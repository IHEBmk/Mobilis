from datetime import date
from django.utils import timezone
from django.utils.timezone import localtime
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.db.models import Avg

from .authentication import CustomJWTAuthentication
from .models import Coordinates, PointOfSale, User,  Visit

class CVIDetailsAPIView(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user  
        
        if user.role == 'admin':
            cvi_users = User.objects.filter(role='agent').select_related('zone')
        elif user.role == 'manager':
            cvi_users = User.objects.filter(role='agent', zone__manager=user).select_related('zone')
        else:
            return Response({"error": "Unauthorized role"}, status=status.HTTP_403_FORBIDDEN)

        data = []
        for cvi in cvi_users:
            zone = cvi.zone

            if zone:
                nb_visits = Visit.objects.filter(agent=cvi, status="finished").count()
                nb_total_visits = Visit.objects.filter(agent=cvi).count()
            else:
                nb_visits = 0
                nb_total_visits = 0

            data.append({
                "id": cvi.id,
                "name": f"{cvi.first_name} {cvi.last_name}",
                "email": cvi.email,
                "phone": cvi.phone,
                "zone_id": zone.id if zone else None,
                "zone_commune": zone.commune if zone else None,
                "status": cvi.status,
                "last_visit": Visit.objects.filter(agent=cvi).order_by('-visit_time').values_list('visit_time', flat=True).first(),
                "nb_visits_finished": nb_visits,
                "nb_total_visits": nb_total_visits
            })

        return Response(data, status=status.HTTP_200_OK)


class CVIProfileAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request, cvi_id):
        try:
            cvi = User.objects.select_related('zone').get(id=cvi_id, role='agent')
            zone = cvi.zone  
            
            last_visit = Visit.objects.filter(agent=cvi).order_by('-visit_time').values_list('visit_time', flat=True).first()
            
            today = date.today()
            visits_today = Visit.objects.filter(
                agent=cvi,
                visit_time__date=today , status="finished"
            ).count()
            
            nb_pdv = Visit.objects.filter(agent__zone=zone).count() if zone else 0
            
            data = {
                "name": f"{cvi.first_name} {cvi.last_name}",
                "email": cvi.email,
                "phone": cvi.phone,
                "zone_name": zone.name if zone else None,
                "status": cvi.status,
                "last_visit": localtime(last_visit) if last_visit else None,
                "nb_visits_today": visits_today,
                "nb_total": nb_pdv
            }
            
            return Response(data, status=200)
        
        except User.DoesNotExist:
            return Response({"error": "CVI not found"}, status=404)
        

class CVILastVisitsAPIView(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, cvi_id):
        try:
            cvi = User.objects.get(id=cvi_id, role='agent')  
        except User.DoesNotExist:
            return Response({"error": "CVI not found"}, status=404)

        visits = Visit.objects.filter(agent=cvi).select_related('agent__zone').order_by('-visit_time')[:5]

        data = []
        for visit in visits:
            zone = visit.agent.zone  
            data.append({
                "visit_id": visit.id,
                "visit_time": localtime(visit.visit_time),
                "duration": visit.duration,
                "zone_commune": zone.commune if zone else None
            })

        return Response(data, status=200)

class CVIVisitsRealizedVsGoal(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, cvi_id):
        try:
            user = request.user

            cvi = User.objects.get(id=cvi_id, role='agent')

            if user.role != 'admin' and (user.role == 'manager' and cvi.zone.manager != user):
                return Response({"error": "Unauthorized role"}, status=403)

            now = timezone.now()
            current_year, current_month = now.year, now.month

            data = []
            for i in range(6):
                month = current_month - i
                year = current_year

                if month <= 0:  
                    month += 12
                    year -= 1

                realized_visits = Visit.objects.filter(
                    agent=cvi,
                    visit_time__year=year,
                    visit_time__month=month, status="finished"
                ).count()

                zone = cvi.zone
                goal_visits = Visit.objects.filter(agent=cvi,
                    visit_time__year=year,
                    visit_time__month=month).count() if zone else 0

                data.append({
                    "year": year,
                    "month": month,
                    "realized_visits": realized_visits,
                    "goal_visits": goal_visits
                })

            return Response(data, status=200)

        except User.DoesNotExist:
            return Response({"error": "CVI not found"}, status=404)
        
class CVIVisitPerformanceAPIView(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, cvi_id):
        try:
            user = request.user

            cvi = User.objects.get(id=cvi_id, role='agent')

            if user.role != 'admin' and (user.role == 'manager' and cvi.zone.manager != user):
                return Response({"error": "Unauthorized role"}, status=403)

            now = timezone.now()
            current_year, current_month = now.year, now.month

            zone = cvi.zone
            total_goal_pdv = Visit.objects.filter(agent=cvi,
                    visit_time__year=current_year,
                    visit_time__month=current_month).values("pdv").distinct().count() if zone else 0

            visited_pdv_this_month = Visit.objects.filter(
                agent=cvi,
                visit_time__year=current_year,
                visit_time__month=current_month, status="finished"
            ).values("pdv").distinct().count()

            total_visited_pdv = Visit.objects.filter(agent=cvi).values("pdv").distinct().count()

            visit_percentage = (total_visited_pdv / total_goal_pdv * 100) if total_goal_pdv else 0

            avg_duration_this_month = Visit.objects.filter(
                agent=cvi,
                visit_time__year=current_year,
                visit_time__month=current_month, status="finished"
            ).aggregate(Avg("duration"))["duration__avg"]

            avg_duration_this_month = round(avg_duration_this_month, 2) if avg_duration_this_month else 0

            return Response({
                "cvi_id": cvi.id,
                "name": f"{cvi.first_name} {cvi.last_name}",
                "visit_percentage": round(visit_percentage, 2),
                "avg_visit_duration": avg_duration_this_month,
                "visited_pdv_this_month": visited_pdv_this_month
            }, status=200)

        except User.DoesNotExist:
            return Response({"error": "CVI not found"}, status=404)
        
class CVICoordinatesAPIView(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, cvi_id):
        try:
            cvi = User.objects.filter(id=cvi_id, role="agent").first()
            if not cvi:
                return Response({"error": "CVI not found"}, status=status.HTTP_404_NOT_FOUND)

            latest_coordinates = Coordinates.objects.filter(user=cvi).order_by("-created_at").first()
            if not latest_coordinates:
                return Response({"error": "No coordinates found for this CVI"}, status=status.HTTP_404_NOT_FOUND)

            data = {
                "cvi_id": cvi.id,
                "name": f"{cvi.first_name} {cvi.last_name}",
                "latitude": latest_coordinates.lattitude,
                "longitude": latest_coordinates.longitude,
                "last_updated": latest_coordinates.created_at,
            }

            return Response(data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
