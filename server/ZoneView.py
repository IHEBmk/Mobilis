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
            
            # Create a lookup dictionary for communes by name
            commune_lookup = {commune['name']: commune['id'] for commune in communes}
        except Wilaya.DoesNotExist:
            return Response({'error': 'Wilaya not found'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Delete existing zones - optimize with prefetch and bulk operations
        # First get the zones to delete in a single query
        existing_zones_query = Zone.objects.filter(pointofsale__commune__in=commune_ids).distinct()
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
        
        # Assign communes using GeoJSON if needed
        if not ('Commune' in df.columns and df['Commune'].notna().all()):
            df = assign_communes_from_geojson(df, 
                "https://qlluxlhcvjnlicxzxwry.supabase.co/storage/v1/object/sign/communes/geoBoundaries-DZA-ADM3.geojson?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJjb21tdW5lcy9nZW9Cb3VuZGFyaWVzLURaQS1BRE0zLmdlb2pzb24iLCJpYXQiOjE3NDQ0NzYzOTMsImV4cCI6MzMyODA0NzYzOTN9.zEbNpeH6f2gp-n3Jrue20cC3cHAq4wZ00Duy6IeEsGs", 
                lat_col, lon_col)
        
        # Optimize point creation - check all existing points in a single query
        errors = []
        created_points = 0
        
        # Get all existing points coordinates in a single query to avoid repeated lookups
        existing_points = set(PointOfSale.objects.filter(
            commune__in=commune_ids
        ).values_list('latitude', 'longitude'))
        
        # Prepare bulk creation lists
        points_to_create = []
        
        for _, row in df.iterrows():
            try:
                lat, lon = row[lat_col], row[lon_col]
                
                # Check if point already exists using our in-memory set
                if (lat, lon) not in existing_points:
                    # Find the commune for this point
                    commune_name = row.get('Commune')
                    commune_id = None
                    if commune_name and commune_name in commune_lookup:
                        commune_id = commune_lookup[commune_name]
                    
                    # Prepare point of sale object for bulk creation
                    points_to_create.append(
                        PointOfSale(
                            id=uuid.uuid4(),
                            latitude=lat,
                            longitude=lon,
                            zone=None,  # Will be set during zoning
                            commune_id=commune_id,  # Use the ID directly to avoid extra query
                            name=f"PDV_{lat}_{lon}",  # Generate a default name
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
            PointOfSale.objects.bulk_create(points_to_create)
        
        # Get all points of sale in this wilaya for zoning - include coordinates in values to avoid lookups later
        pdvs_to_zone = list(PointOfSale.objects.filter(commune__in=commune_ids).values(
            'id', 'latitude', 'longitude', 'commune_id'
        ))
        if not pdvs_to_zone:
            return Response({'error': 'No points of sale found for this wilaya after importing'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create a lookup for PDVs by coordinates for faster access later
        pdv_lookup = {(pdv['latitude'], pdv['longitude']): pdv['id'] for pdv in pdvs_to_zone if pdv['latitude'] and pdv['longitude']}
        
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

        # Create balanced zones
        df, zones, zone_workloads, zone_communes = create_balanced_zones(
            df, number_of_zones, lat_col, lon_col, 
            points_coef=points_coef, distance_coef=distance_coef
        )

        # Generate zone boundaries
        zone_polygons = generate_zone_boundaries(zones)

        # Get available managers for assignment - do this in a single query
        available_managers = list(User.objects.filter(role='manager', wilaya=wilaya_id))
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
                    lat, lon = row[lat_col], row[lon_col]
                    # Look up using the tuple key properly
                    pdv_id = pdv_lookup.get((lat, lon))
                    
                    if pdv_id:
                        # Add to our list of updates
                        pdv_zone_updates.append((pdv_id, zone_uuid))
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
        
        # Update PDVs with zones using Django's bulk_update
        if pdv_zone_updates:
            # Get all the PDVs that need updating in a single query
            pdv_ids = [pdv_id for pdv_id, _ in pdv_zone_updates]
            pdvs_to_update = {pdv.id: pdv for pdv in PointOfSale.objects.filter(id__in=pdv_ids)}
            
            # Set the zone for each PDV
            for pdv_id, zone_id in pdv_zone_updates:
                if pdv_id in pdvs_to_update:
                    pdvs_to_update[pdv_id].zone_id = zone_id
            
            # Perform bulk update
            PointOfSale.objects.bulk_update(pdvs_to_update.values(), ['zone_id'])
        
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
            zones = Zone.objects.all()
        elif user.role == 'manager':
            supervised_agents = User.objects.filter(manager=user.id).values()
            zones = list(Zone.objects.filter(manager__in=supervised_agents).values())
        else:
            return Response({'error': 'you can\'t retrieve zones'}, status=status.HTTP_401_UNAUTHORIZED)
        for zone in zones:
            zone['manager']=zone['manager'].first_name
            zone['wilaya']=zone['commune'].wilaya.name
            pdvs=PointOfSale.objects.filter(zone=zone['id'])
            zone['pdvs']=len(pdvs)
            visists=Visit.objects.filter(pdv__in=pdvs)
            sheduled_visits=len(visists.filter(status='scheduled'))
            total_visits=len(pdvs)
            zone['couverture']=round(sheduled_visits/total_visits*100,2)
        return Response({'zones': zones}, status=status.HTTP_200_OK)

        




