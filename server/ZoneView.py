import json
from datetime import datetime, timezone
import random
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import pandas as pd
from server.authentication import CustomJWTAuthentication

from server.ZoneSerializer import ZoneSerializer
from server.models import Commune, PointOfSale, User, Wilaya,Zone,Visit
from rest_framework.permissions import IsAuthenticated
from server.zoning import assign_communes_from_geojson, create_balanced_zones, export_zones_to_geojson, generate_zone_boundaries, load_data
import json
from datetime import datetime
import random
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import pandas as pd
from server.authentication import CustomJWTAuthentication

from server.ZoneSerializer import ZoneSerializer
from server.models import Commune, PointOfSale, User, Wilaya,Zone,Visit
from rest_framework.permissions import IsAuthenticated
from server.zoning import assign_communes_from_geojson, create_balanced_zones, export_zones_to_geojson, generate_zone_boundaries, load_data






class GenerateZones(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        user = request.user
        wilaya_id = request.data.get('wilaya_id')
        number_of_zones = int(request.data.get('number_of_zones'))
        csv_file = request.FILES.get('csv_file')
        lat_col = request.data.get('lat_col', 'Latitude')
        lon_col = request.data.get('lon_col', 'Longitude')
        balance_type = request.data.get('balance_type', 'balanced')

        # Validate required parameters
        if not csv_file:
            return Response({'error': 'CSV file is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        if user.role == 'admin':
            if not wilaya_id:
                return Response({'error': 'Wilaya ID is required for admin users'}, status=status.HTTP_400_BAD_REQUEST)
        elif user.role == 'manager':
            if not user.wilaya:
                return Response({'error': 'Manager does not have an assigned wilaya'}, status=status.HTTP_400_BAD_REQUEST)
            wilaya_id = str(user.wilaya.id)
        else:
            return Response({'error': 'You do not have permission to generate zones'}, status=status.HTTP_403_FORBIDDEN)
        
        # Get the wilaya and communes in a single database interaction
        try:
            wilaya = Wilaya.objects.select_related().get(id=wilaya_id)
            # Fetch all communes in a single query and store in memory
            communes = list(Commune.objects.filter(wilaya=wilaya_id).values())
            commune_ids = [commune['id'] for commune in communes]
            
            # Create a lookup dictionary for communes by name - normalize names for case-insensitive comparison
            commune_lookup = {commune['name'].lower().strip(): commune['id'] for commune in communes}
            # Also create reverse lookup for ID to name
            commune_id_to_name = {commune['id']: commune['name'] for commune in communes}
            print(f"Available communes in wilaya {wilaya.name}: {list(commune_lookup.keys())}")
        except Wilaya.DoesNotExist:
            return Response({'error': 'Wilaya not found'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Delete existing zones - optimize with prefetch and bulk operations
        # First get the zones to delete in a single query
        existing_zones_query = Zone.objects.filter(manager__wilaya_id=wilaya_id).distinct()
        zones_count = existing_zones_query.count()
        
        if zones_count > 0:
            # Update all points of sale in these zones to have zone=None in a single bulk update
            zone_ids = list(existing_zones_query.values_list('id', flat=True))
            PointOfSale.objects.filter(zone__in=zone_ids).update(zone=None)
            
            # Now delete the zones in a single operation
            deletion_count = existing_zones_query.delete()[0]
            print(f"Deleted {deletion_count} existing zones in wilaya {wilaya.name}")
        
        # Load data from CSV
        df = load_data(csv_file, lat_col, lon_col)
        print(f"Loaded {len(df)} rows from CSV. Columns: {df.columns.tolist()}")
        
        # Assign communes using GeoJSON if needed
        if not ('Commune' in df.columns and df['Commune'].notna().all()):
            print(f"Assigning communes from GeoJSON as 'Commune' column is missing or has NULL values")
            df = assign_communes_from_geojson(df, 
                "https://qlluxlhcvjnlicxzxwry.supabase.co/storage/v1/object/sign/communes/geoBoundaries-DZA-ADM3.geojson?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJjb21tdW5lcy9nZW9Cb3VuZGFyaWVzLURaQS1BRE0zLmdlb2pzb24iLCJpYXQiOjE3NDQ0NzYzOTMsImV4cCI6MzMyODA0NzYzOTN9.zEbNpeH6f2gp-n3Jrue20cC3cHAq4wZ00Duy6IeEsGs", 
                lat_col, lon_col)
            print(f"After GeoJSON assignment, commune distribution: {df['Commune'].value_counts().to_dict()}")
        else:
            print(f"Using provided communes. Commune distribution: {df['Commune'].value_counts().to_dict()}")
        
        # Optimize point creation - check all existing points in a single query
        errors = []
        created_points = 0
        
        # Get all existing points with full details for renaming
        existing_points_data = list(PointOfSale.objects.filter(
            commune__in=commune_ids
        ).values('id', 'latitude', 'longitude', 'commune_id'))
        
        existing_points_coords = set((float(p['latitude']), float(p['longitude'])) for p in existing_points_data)
        print(f"Found {len(existing_points_data)} existing points of sale in this wilaya")
        
        # Rename all existing PDVs to follow the sequential naming convention
        existing_pdvs_to_rename = []
        commune_counters = {}  # Track current counter for each commune
        
        # Group existing PDVs by commune for sequential renaming
        existing_pdvs_by_commune = {}
        for pdv in existing_points_data:
            commune_id = pdv['commune_id']
            if commune_id:
                commune_name = commune_id_to_name.get(commune_id, f'commune_{commune_id}')
                if commune_name not in existing_pdvs_by_commune:
                    existing_pdvs_by_commune[commune_name] = []
                existing_pdvs_by_commune[commune_name].append(pdv)
        
        # Rename existing PDVs with sequential numbers
        for commune_name, pdvs in existing_pdvs_by_commune.items():
            commune_counters[commune_name] = 0
            for pdv in pdvs:
                commune_counters[commune_name] += 1
                new_name = f"{commune_name}__{commune_counters[commune_name]}"
                existing_pdvs_to_rename.append({
                    'id': pdv['id'],
                    'new_name': new_name
                })
        
        # Bulk update existing PDV names
        if existing_pdvs_to_rename:
            pdvs_to_update_names = {pdv.id: pdv for pdv in PointOfSale.objects.filter(
                id__in=[p['id'] for p in existing_pdvs_to_rename]
            )}
            
            for rename_data in existing_pdvs_to_rename:
                pdv_id = rename_data['id']
                if pdv_id in pdvs_to_update_names:
                    pdvs_to_update_names[pdv_id].name = rename_data['new_name']
            
            PointOfSale.objects.bulk_update(pdvs_to_update_names.values(), ['name'])
            print(f"Renamed {len(existing_pdvs_to_rename)} existing PDVs with sequential naming")
        
        print(f"Current PDV counts per commune after renaming: {commune_counters}")
        
        # Map commune names to IDs, handling potential case/spacing differences
        def get_commune_id(commune_name):
            if not commune_name:
                return None
            normalized_name = commune_name.lower().strip()
            return commune_lookup.get(normalized_name)
        
        # Check if there are any communes in the CSV that don't match our lookup
        unmapped_communes = set()
        for commune_name in df['Commune'].unique():
            if commune_name and get_commune_id(commune_name) is None:
                unmapped_communes.add(commune_name)
        
        if unmapped_communes:
            print(f"WARNING: {len(unmapped_communes)} commune names in CSV don't match database: {unmapped_communes}")
        
        # Prepare bulk creation lists
        points_to_create = []
        for _, row in df.iterrows():
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                # Check if point already exists using our in-memory set
                if (lat, lon) not in existing_points_coords:
                    # Find the commune for this point
                    commune_name = row.get('Commune')
                    commune_id = get_commune_id(commune_name)
                    
                    # Record if we couldn't map this commune
                    if commune_name and not commune_id:
                        errors.append({
                            'type': 'commune_mapping_error',
                            'commune': commune_name,
                            'latitude': lat,
                            'longitude': lon,
                            'error': f"Commune '{commune_name}' not found in wilaya {wilaya.name}"
                        })
                        # Use a default name for unmapped communes
                        pdv_name = f"Unknown_Commune__{lat}_{lon}"
                    else:
                        # Generate sequential name for the commune (continuing from existing counter)
                        if commune_name not in commune_counters:
                            commune_counters[commune_name] = 0
                        commune_counters[commune_name] += 1
                        pdv_name = f"{commune_name}__{commune_counters[commune_name]}"
                    
                    # Prepare point of sale object for bulk creation
                    # NOTE: manager will be set later after zone assignment
                    points_to_create.append(
                        PointOfSale(
                            id=uuid.uuid4(),
                            latitude=lat,
                            manager=None,  # Will be set to zone manager later
                            longitude=lon,
                            zone=None,  # Will be set during zoning
                            commune_id=commune_id,  # Use the ID directly to avoid extra query
                            name=pdv_name,  # Use sequential naming format
                            created_at=datetime.now(timezone.utc),  # Use timezone-aware datetime
                            status=1,  # Default status (active)
                        )
                    )
            except Exception as e:
                errors.append({
                    'type': 'pdv_creation_error',
                    'latitude': row[lat_col],
                    'longitude': row[lon_col],
                    'error': str(e)
                })
        
        # Bulk create all new points in a single database operation
        if points_to_create:
            created_points = len(points_to_create)
            print(f"Creating {created_points} new points of sale")
            PointOfSale.objects.bulk_create(points_to_create)
            print(f"Successfully created {created_points} points of sale")
            
            # Log the final counts per commune
            print(f"Final PDV counts per commune: {commune_counters}")
        else:
            print("No new points to create")
        
        # If no points were created but some were mapped to invalid communes, provide a better error
        if created_points == 0 and any(e['type'] == 'commune_mapping_error' for e in errors):
            return Response({
                'error': 'No points of sale could be created because the communes in the CSV do not match the communes in the selected wilaya',
                'details': {
                    'unmapped_communes': list(unmapped_communes),
                    'available_communes': list(commune_lookup.keys())
                },
                'errors': errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get all points of sale in this wilaya for zoning - include coordinates in values to avoid lookups later
        pdvs_to_zone = list(PointOfSale.objects.filter(commune__in=commune_ids).values(
            'id', 'latitude', 'longitude', 'commune_id'
        ))
        print(f"Found {len(pdvs_to_zone)} points of sale to zone")
        
        if not pdvs_to_zone:
            return Response({
                'error': 'No points of sale found for this wilaya after importing',
                'details': {
                    'created_points': created_points,
                    'wilaya_id': wilaya_id,
                    'commune_count': len(communes),
                    'commune_ids': commune_ids
                },
                'errors': errors
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create a lookup for PDVs by coordinates for faster access later
        pdv_lookup = {(float(pdv['latitude']), float(pdv['longitude'])): pdv['id'] for pdv in pdvs_to_zone if pdv['latitude'] and pdv['longitude']}
        
        # Set balance coefficients based on balance_type
        if balance_type == "points":
            points_coef = 10
            distance_coef = 0.1
        elif balance_type == "distance":
            points_coef = 0.1
            distance_coef = 10
        else:
            points_coef = 1
            distance_coef = 1
        
        # Prepare data for zoning algorithm
        zoning_df = pd.DataFrame({
            'Latitude': [float(pdv['latitude']) for pdv in pdvs_to_zone],
            'Longitude': [float(pdv['longitude']) for pdv in pdvs_to_zone]
        })
        
        if 'Commune' in df.columns:
            # Try to map commune IDs back to names for zoning
            commune_names = []
            for pdv in pdvs_to_zone:
                commune_id = pdv.get('commune_id')
                commune_names.append(commune_id_to_name.get(commune_id, 'Unknown'))
            zoning_df['Commune'] = commune_names
        
        print(f"Starting zone creation with {len(zoning_df)} points")
        df, zones, zone_workloads, zone_communes = create_balanced_zones(
            zoning_df, number_of_zones, lat_col, lon_col, 
            points_coef=points_coef, distance_coef=distance_coef
        )
        print(f"Created {len(zones)} zones")

        # Generate zone boundaries
        zone_polygons = generate_zone_boundaries(zones)

        # Get available managers for assignment - do this in a single query
        available_managers = list(User.objects.filter(role='agent', wilaya=wilaya_id))
        print(f'Found {len(available_managers)} available managers')
        manager_count = len(available_managers)
        
        if manager_count < number_of_zones:
            # Not enough managers, we'll reuse some
            manager_assignments = random.choices(available_managers, k=number_of_zones)
        else:
            # We have enough managers, randomly select without replacement
            manager_assignments = random.sample(available_managers, number_of_zones)
        
        # Map zone_id to manager
        zone_managers = dict(zip([str(i) for i in zones.keys()], manager_assignments))
        
        # Create zones in the database with bulk operations
        created_zones = {}
        zones_to_create = []
        pdv_zone_updates = []
        pdv_manager_updates = []  # NEW: Track manager updates for PDVs
        
        # Prepare zone creation data
        for zone_id, zone_df in zones.items():
            try:
                zone_id_str = str(zone_id)
                manager = zone_managers.get(zone_id_str)
                
                # Format zone name
                name = f"{wilaya.name}_{zone_id}"
                
                # Create zone object
                zone_uuid = uuid.uuid4()
                created_at = datetime.now(timezone.utc)  # Use timezone-aware datetime
                
                # Create zone object for bulk creation
                new_zone = Zone(
                    id=zone_uuid,
                    created_at=created_at,
                    commune=None,
                    name=name,
                    manager_id=manager.id if manager else None
                )
                
                zones_to_create.append(new_zone)
                created_zones[zone_id] = new_zone
                
                # Collect PDV updates for this zone
                for _, row in zone_df.iterrows():
                    lat, lon = float(row[lat_col]), float(row[lon_col])
                    # Look up using the tuple key properly
                    pdv_id = pdv_lookup.get((lat, lon))
                    if pdv_id:
                        # Add to our list of updates
                        pdv_zone_updates.append((pdv_id, zone_uuid))
                        # NEW: Also track manager assignment
                        if manager:
                            pdv_manager_updates.append((pdv_id, manager.id))
                    else:
                        errors.append({
                            'type': 'pdv_update_error',
                            'zone_id': zone_id,
                            'latitude': lat,
                            'longitude': lon,
                            'error': 'PDV not found at coordinates'
                        })
                        
            except Exception as e:
                errors.append({
                    'type': 'zone_creation_error',
                    'zone_id': zone_id,
                    'error': str(e)
                })
        
        # Bulk create all zones in a single database operation
        Zone.objects.bulk_create(zones_to_create)
        print(f"Created {len(zones_to_create)} zones in database")
        
        # Update PDVs with zones and managers using Django's bulk_update
        if pdv_zone_updates:
            # Get all the PDVs that need updating in a single query
            pdv_ids = [pdv_id for pdv_id, _ in pdv_zone_updates]
            pdvs_to_update = {pdv.id: pdv for pdv in PointOfSale.objects.filter(id__in=pdv_ids)}
            
            # Create lookup for manager assignments
            pdv_manager_lookup = dict(pdv_manager_updates)
            
            # Set the zone and manager for each PDV
            for pdv_id, zone_id in pdv_zone_updates:
                if pdv_id in pdvs_to_update:
                    pdvs_to_update[pdv_id].zone_id = zone_id
                    # NEW: Also set the manager to match the zone
                    if pdv_id in pdv_manager_lookup:
                        pdvs_to_update[pdv_id].manager_id = pdv_manager_lookup[pdv_id]
            
            # Perform bulk update - NOW INCLUDING MANAGER
            PointOfSale.objects.bulk_update(pdvs_to_update.values(), ['zone', 'manager'])
            print(f"Updated {len(pdvs_to_update)} points of sale with zone and manager assignments")
        
        # Export zones to GeoJSON and update wilaya
        try:
            # Generate GeoJSON directly as a dictionary (no file output)
            geojson_data = export_zones_to_geojson(
                df, zones, zone_workloads, zone_communes, zone_polygons,
                output_path=None,  # Don't write to file
                commune_geojson="https://qlluxlhcvjnlicxzxwry.supabase.co/storage/v1/object/sign/communes/geoBoundaries-DZA-ADM3.geojson?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJjb21tdW5lcy9nZW9Cb3VuZGFyaWVzLURaQS1BRE0zLmdlb2pzb24iLCJpYXQiOjE3NDQ0NzYzOTMsImV4cCI6MzMyODA0NzYzOTN9.zEbNpeH6f2gp-n3Jrue20cC3cHAq4wZ00Duy6IeEsGs"
            )
            
            # Update wilaya with GeoJSON data
            if isinstance(geojson_data, dict):
                wilaya.geojson = json.dumps(geojson_data)
                wilaya.save()
                print(f"Successfully saved GeoJSON to wilaya {wilaya.name}")
            else:
                errors.append({
                    'type': 'geojson_generation_error',
                    'error': "Expected dict result from export_zones_to_geojson"
                })
        except Exception as e:
            errors.append({
                'type': 'geojson_save_error',
                'error': str(e)
            })
        
        response_data = {
            'success': True,
            'points_created': created_points,
            'zones_created': len(created_zones),
            'zones_deleted': zones_count,
            'errors': errors if errors else None
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
                
        
        


class GetGeojsonWilaya(APIView):
    def post(self, request):
        id = request.data.get('id')
        wilaya_id=request.data.get('wilaya_id')
        if not id or not wilaya_id:
            return Response({'error': 'ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=id)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager' and user.role!='admin':
            return Response({'error': 'you can\'t retrieve zones'}, status=status.HTTP_401_UNAUTHORIZED)
        if user.role=='manager':
            if wilaya_id!=str(user.wilaya.id):
                return Response({'error': 'you can\'t generate zones'}, status=status.HTTP_400_BAD_REQUEST)
        geosjon_txt=user.wilaya.geojson
        geosjon=json.loads(geosjon_txt)
        return Response({'message': 'Geojson generated successfully',
                         'geojson': geosjon}, status=status.HTTP_200_OK)

class GetZones(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user = request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role == 'admin':
            zones = list(Zone.objects.all().values())
            
        elif user.role == 'manager':
            supervised_agents = User.objects.filter(manager=user.id).values_list('id', flat=True)
            zones = list(Zone.objects.filter(manager__in=supervised_agents).values())
        else:
            return Response({'error': 'you can\'t retrieve zones'}, status=status.HTTP_401_UNAUTHORIZED)
        for zone in zones:
            print(zone)
            manager=User.objects.get(id=zone['manager_id'])
            zone['manager']=manager.first_name
            try:
                
                zone['wilaya']=manager.wilaya.name
            except:
                zone['wilaya']='None'
            pdvs=PointOfSale.objects.filter(zone=zone['id'])
            zone['pdvs']=len(pdvs)
            visists=Visit.objects.filter(pdv__in=pdvs)
            sheduled_visits=len(visists.filter(status='scheduled'))
            total_visits=len(pdvs)
            zone['couverture']=round(sheduled_visits/total_visits*100,2)
        return Response({'zones': zones}, status=status.HTTP_200_OK)
        


