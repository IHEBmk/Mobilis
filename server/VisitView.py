import base64
from collections import defaultdict
from datetime import datetime, timedelta
import json
from typing import Dict
from django.utils import timezone
import uuid
from django.forms import model_to_dict
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pandas as pd
from server.Visit_paling_model import calculate_flexible_statistics, create_schedule_from_deadline, plan_multiple_days_flexible
from server.authentication import CustomJWTAuthentication
from server.models import  PointOfSale, User, Visit
from rest_framework.permissions import IsAuthenticated
from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.timezone import make_aware



class GetVisits(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role=='admin':
            visits=Visit.objects.filter(validated=1)
        elif user.role=='manager':
            supervised_agents=User.objects.filter(manager=user.id).values()
            visits=Visit.objects.filter(agent__in=supervised_agents,validated=1)
        elif user.role=='agent':
            visits=Visit.objects.filter(agent=user.id,status='scheduled',validated=1)
        else:
            return Response({'error': 'you can\'t retrieve visits'}, status=status.HTTP_401_UNAUTHORIZED)
        visits_list=[]
        for visit in visits:
            pdv=visit.pdv
            visits_list.append({'id':visit.id,'pdv':pdv.name,'deadline':visit.deadline})
        return Response({'visits': visits_list}, status=status.HTTP_200_OK)
    
    
    
class VisitPdv(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        remake = data.get('remake')
        speed_kmph = float(data.get('speed_kmph', 60))
        
        if not data.get('pdv'):
            return Response({'error': 'Missing pdv or id'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = request.user
            if not user:
                return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)

            pdv = PointOfSale.objects.get(id=data.get('pdv'))
        except (User.DoesNotExist, PointOfSale.DoesNotExist):
            return Response({'error': 'User or Point of Sale not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.role != "agent":
            return Response({'error': "You are not an agent"}, status=status.HTTP_403_FORBIDDEN)

        try:
            # Get the visit that is scheduled for the Point of Sale and agent
            visit = Visit.objects.get(pdv=pdv, agent=user, status='scheduled')

            # Check if the visit deadline is tomorrow or later
            if visit.deadline.date() > timezone.now().date():
                # Get today's date in the local timezone
                today = timezone.localdate()

                # Filter visits that are scheduled for today and not yet completed
                today_visits = Visit.objects.filter(agent=user, validated=1, status='scheduled', deadline__date=today)

                # If there are any scheduled visits for today, return an error
                if today_visits.exists():
                    min_order = visit.order
                    for visit in today_visits:
                        min_order = min(min_order, visit.order)
                    if visit.order >= min_order:
                        return Response({'error': 'You can\'t visit non-scheduled visits until you finish the current scheduled visits'}, 
                                    status=status.HTTP_400_BAD_REQUEST)
            visit.status = 'visited'
            visit.visit_time=timezone.now()
            visit.save()

            pdv.last_visit = timezone.now()
            pdv.save()
            print(remake)
            if visit.deadline.date() == timezone.now().date():
                return Response({'message': "Visited successfully"}, status=status.HTTP_200_OK)

            if int(remake)==1:
                print(f"Remake")
                print(f"Remake: {remake}")
                now = timezone.now()+timedelta(days=1)
                visits = Visit.objects.filter(agent=user, validated=1, status='scheduled', deadline__gte=now)
                data_points = [
                    {'id': visit.pdv.id, 'name': visit.pdv.name, 'longitude': visit.pdv.longitude, 'latitude': visit.pdv.latitude}
                    for visit in visits
                ]
                start_point = {
                    'name': "Start",
                    'longitude': user.agence.longitude,
                    'latitude': user.agence.latitude
                }

                points=[]
                for visit in visits:
                    points.append(visit.pdv.id)
                    visit.delete()
                deadline=user.deadline
                future_start = now
                schedule_map= create_schedule_from_deadline(
                    start_date=future_start.strftime('%Y-%m-%d'),
                    deadline_date=deadline.strftime('%Y-%m-%d'),
                    weekday_minutes=480, 
                    weekend_minutes=0     
                )
                # Assuming you are planning again with all PDVs managed by user
                data_points = list(PointOfSale.objects.filter(manager=user,id__in=points).values())
                routes, edges, estimates,schedule_used = plan_multiple_days_flexible([start_point] + data_points, schedule_map, float(speed_kmph))
                new_visits = []
                for day_index, route in enumerate(routes):
                    for order, point_name in enumerate(route[1:], start=1):  # skip start point
                        pos = next((p for p in data_points if p['name'] == point_name), None)
                        if not pos:
                            continue
                        new_visit = Visit(
                            id=uuid.uuid4(),
                            deadline=future_start + timedelta(days=day_index),
                            agent=user,
                            pdv_id=pos['id'],
                            status='scheduled',
                            order=order,
                            validated=1
                        )
                        new_visits.append(new_visit)

                Visit.objects.bulk_create(new_visits)
                return Response({
                    'message': 'Visits rescheduled successfully',
                    
                }, status=status.HTTP_201_CREATED)

        except Visit.DoesNotExist:
            return Response({'error': 'Visit was not scheduled'}, status=status.HTTP_400_BAD_REQUEST)
        
    
    
    
    

    
class MakePlanning(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        
        data = request.data
        user = request.user
        schedule_map = data.get('mapping')
        speed_kmph = float(data.get('speed_kmph', 60))
        remake = data.get('remake')
        cvi_id = data.get('cvi')
        deadline_str = data.get('deadline')
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'manager' and user.role != 'admin':
            return Response({'error': 'You can\'t make planning your are an agent'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not deadline_str or not cvi_id:
            return Response({'error': 'Deadline and CVI are required'}, status=status.HTTP_400_BAD_REQUEST)
        if not schedule_map or not isinstance(schedule_map, dict):
             schedule_map = create_schedule_from_deadline(
        start_date= datetime.now().strftime('%Y-%m-%d'),
        deadline_date= deadline_str,
        weekday_minutes=480, 
        weekend_minutes=0     
    )
        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%d')
        except:
            return Response({'error': 'Deadline format invalid'}, status=status.HTTP_400_BAD_REQUEST)

        now = datetime.now()
        if deadline.date() < now.date():
            return Response({'error': 'Deadline is in the past'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            cvi = User.objects.get(id=cvi_id)
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user and user.role != 'admin':
            return Response({'error': 'You are not authorized to make planning'}, status=status.HTTP_403_FORBIDDEN)
        if int(remake)==1:
            Visit.objects.filter(agent=cvi, validated=0).delete()


        if not cvi.agence:
            return Response({'error': 'CVI has no agence assigned'}, status=status.HTTP_400_BAD_REQUEST)
        longitude = cvi.agence.longitude if not None else 31.610709890371677
        latitude = cvi.agence.latitude if not None else 2.8828559973535572
        start_point = {
            'name': 'Start',
            'longitude':longitude,
            'latitude':latitude
        }

        data_points = list(PointOfSale.objects.filter(manager=cvi).values('id', 'name', 'longitude', 'latitude'))
        routes, edges, estimates,schedule_used = plan_multiple_days_flexible(
            [start_point] + data_points,
            schedule_map,
            float(speed_kmph)
        )
        
        stats = calculate_flexible_statistics(routes, estimates,schedule_used, [start_point] + data_points)
        warning = None
        if len(schedule_map.keys())<len(schedule_used.keys()):
            warning = "The CVI won't be able to visit all the points"
            suggestion='Increase the deadline'
            return Response({
            'message': 'Visits could not be scheduled',
            'number_of_days': len(schedule_map.keys()),
            'estimated_days':len(schedule_used.keys()),
            'number_of_points':len(data_points),
            'suggestion':suggestion,
            'warning': warning
        }, status=status.HTTP_400_BAD_REQUEST)
        visits = []
        for day_index, route in enumerate(routes):
            for order_index, point_name in enumerate(route):
                if point_name=="Start":
                    route[0]=cvi.agence.name if cvi.agence else 'Agence'
                    continue# skip 'Start'
                pos = next((p for p in data_points if p['name'] == point_name), None)
                if pos:
                    visits.append(Visit(
                        id=uuid.uuid4(),
                        deadline=now + timedelta(days=day_index),
                        agent=cvi,
                        pdv_id=pos['id'],
                        status='scheduled',
                        order=order_index,
                        validated=0
                    ))

        Visit.objects.bulk_create(visits)
        cvi.deadline=deadline
        cvi.save()
        return Response({
            'message': 'Visits scheduled successfully',
            'stats': stats,
            'number_of_points': len(data_points),
            'visits':schedule_used,
            'warning': warning
        }, status=status.HTTP_201_CREATED)

        
        
class ValidatePlanning(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data=request.data
        if not data.get('cvi'):
            return Response({'error': 'missing cvi'}, status=status.HTTP_400_BAD_REQUEST)
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager' and user.role!='admin':
            return Response({'error': 'you can\'t validate visits'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            cvi = User.objects.get(id=data.get('cvi'))
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user and user.role != 'admin':
            return Response({'error': 'You are not authorized to make planning'}, status=status.HTTP_403_FORBIDDEN)
        visits=Visit.objects.filter(agent=data.get('cvi'),validated=0)
        deadline=datetime.now()
        for visit in visits:
            if visit.deadline.replace(tzinfo=None) > deadline.replace(tzinfo=None):
                deadline=visit.deadline
            visit.validated=1
            visit.save()
        cvi.deadline=deadline
        cvi.save()
        return Response({'message': 'Visits validated successfully'}, status=status.HTTP_201_CREATED)
    
    
class GetOldPlanning(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        cvi= request.query_params.get('cvi', None)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'manager' and user.role != 'admin':
            return Response({'error': 'You can\'t get planning'}, status=status.HTTP_400_BAD_REQUEST)
        if not cvi:
            return Response({'error': 'Missing CVI'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            cvi = User.objects.get(id=cvi)
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user and user.role != 'admin':
            return Response({'error': 'You are not authorized to get planning'}, status=status.HTTP_403_FORBIDDEN)
        visits = Visit.objects.filter(agent=cvi, validated=0).order_by('deadline', 'order').select_related('pdv')
        if not visits.exists():
            return Response({'visits': None}, status=status.HTTP_200_OK)

        # Group PDV names by deadline date
        visits_by_day = defaultdict(list)
        for visit in visits:
            day = visit.deadline.date().isoformat()  # e.g., '2025-06-01'
            pdv_name = visit.pdv.name
            visits_by_day[day].append(pdv_name)

        # Convert defaultdict to regular dict before returning
        return Response({'visits': dict(visits_by_day)}, status=status.HTTP_200_OK)
    
    
class GetVisitsPlan(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'agent':
            return Response({'error': 'You can\'t get planning'}, status=status.HTTP_400_BAD_REQUEST)

        today = datetime.now().date()
        start_of_day = make_aware(datetime.combine(today, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(today, datetime.max.time()))

        visits = Visit.objects.filter(
            agent=user,
            validated=1,
            status='scheduled',
            deadline__gte=start_of_day,
        ).order_by('deadline', 'order').select_related('pdv')

        # Group by deadline.date()
        grouped = defaultdict(list)
        for visit in visits:
            date_str = visit.deadline.date().isoformat()
            grouped[date_str].append({
                'visit_id': visit.id,
                'order': visit.order,
                'visit_time': visit.visit_time,
                'point_of_sale': {
                    'id': visit.pdv.id,
                    'name': visit.pdv.name,
                    'latitude': visit.pdv.latitude,
                    'longitude': visit.pdv.longitude
                }
            })

        # Prepare final sorted response
        sorted_data = []
        for date in sorted(grouped.keys()):
            sorted_data.append({
                'date': date,
                'visits': sorted(grouped[date], key=lambda x: x['order'])
            })

        return Response({'plan': sorted_data, 'id': user.id}, status=status.HTTP_200_OK)
        
        
    
    
    


class ClancelVisit(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        image=request.FILES.get('image')
        user = request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'agent':
            return Response({'error': 'You can\'t cancel visit'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('visit_id'):
            return Response({'error': 'Missing visit_id'}, status=status.HTTP_400_BAD_REQUEST)
        if not image:
            return Response({'error': 'Missing image'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            visit = Visit.objects.get(id=data.get('visit_id'))
        except Visit.DoesNotExist:
            return Response({'error': 'Visit not found'}, status=status.HTTP_400_BAD_REQUEST)
        if visit.agent != user:
            return Response({'error': 'You are not authorized to cancel visit'}, status=status.HTTP_400_BAD_REQUEST)
        if visit.status != 'scheduled':
            return Response({'error': 'Visit is not scheduled'}, status=status.HTTP_400_BAD_REQUEST)
        visit.status = 'cancelled'
        image_encoded= base64.b64encode(image.read())
        visit.cancel_proof=image_encoded
        visit.save()
        # reschedueling
    #     work_minutes = data.get('work_minutes', 0)
    #     now = timezone.now()
    #     deadline=now
    #     visits=Visit.objects.filter(agent=user,status='scheduled',deadline__gte=now)
    #     data_points=[]
    #     pdv=visit.pdv
    #     data_points+=[{'id':pdv.id,'name':pdv.name,'longitude':pdv.longitude,'latitude':pdv.latitude}]
    #     for visit in visits:
    #         pdv=visit.pdv
    #         data_points.append({'id':pdv.id,'name':pdv.name,'longitude':pdv.longitude,'latitude':pdv.latitude})
    #         deadline=max(deadline,visit.deadline)
    #         visit.delete()
        
    #     longitude = user.agence.longitude if not None else 31.610709890371677
    #     latitude = user.agence.latitude if not None else 2.8828559973535572
    #     start_point = {
    #         'name': 'Start',
    #         'longitude':longitude,
    #         'latitude':latitude
    #     }
    #     schedule_map = create_schedule_from_deadline(
    #     start_date= datetime.now().strftime('%Y-%m-%d'),
    #     deadline_date= deadline.strftime('%Y-%m-%d'),
    #     weekday_minutes=480, 
    #     weekend_minutes=0     
    # )   
    #     schedule_map[datetime.now().strftime('%Y-%m-%d')]=480-work_minutes
    #     number_of_days=(deadline-now).days
    #     daily_limit_minutes=7*60
    #     total_time_minutes=number_of_days*daily_limit_minutes
        
    #     routes,edges,estimates=plan_multiple_days_flexible([start_point]+data_points,total_time_minutes,schedule_map,60)
    #     visits = []
    #     for day_index, route in enumerate(routes):
    #         for order_index, point_name in enumerate(route[1:], start=1):  # skip 'Start'
    #             pos = next((p for p in data_points if p['name'] == point_name), None)
    #             if pos:
    #                 visits.append(Visit(
    #                     id=uuid.uuid4(),
    #                     deadline=now + timedelta(days=day_index+1),
    #                     agent=user,
    #                     pdv_id=pos['id'],
    #                     status='scheduled',
    #                     order=order_index,
    #                     validated=1
    #                 ))

    #     Visit.objects.bulk_create(visits)
        return Response({'message': 'Visit cancelled successfully'}, status=status.HTTP_201_CREATED)
        
        
        
        
class GetCancelledVisits(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user=request.user
        cvi=request.query_params.get('cvi', None)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager' and user.role!='admin':
            return Response({'error': 'you can\'t retrieve visits'}, status=status.HTTP_401_UNAUTHORIZED)
        if not cvi:
            return Response({'error': 'Missing CVI'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            cvi = User.objects.get(id=cvi)
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user and user.role != 'admin':
            return Response({'error': 'You are not authorized to get planning'}, status=status.HTTP_403_FORBIDDEN)
        visits=Visit.objects.filter(agent=cvi,status='cancelled',validated=1).order_by('deadline', 'order').select_related('pdv')
        if not visits.exists():
            return Response({'visits': None}, status=status.HTTP_200_OK)

        # Group PDV names by deadline date
        visits_by_day = defaultdict(list)
        for visit in visits:
            day = visit.deadline.date().isoformat()
            visit = {'id':visit.id,'pdv':visit.pdv.name,'cancel_proof':visit.cancel_proof}
            visits_by_day[day].append(visit)

        # Convert defaultdict to regular dict before returning
        return Response({'visits': dict(visits_by_day)}, status=status.HTTP_200_OK)
    
    
    
class HandleCancelVisit(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        user = request.user
        action=data.get('action')
        visit_id=data.get('visit_id')
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager' and user.role!='admin':
            return Response({'error': 'you can\'t retrieve visits'}, status=status.HTTP_401_UNAUTHORIZED)
        if not action:
            return Response({'error': 'Missing action'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            cvi = User.objects.get(id=data.get('cvi'))
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user and user.role != 'admin':
            return Response({'error': 'You are not authorized to get planning'}, status=status.HTTP_403_FORBIDDEN)
        if not visit_id:
            return Response({'error': 'Missing visit_id'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            visit = Visit.objects.get(id=visit_id)
            if visit.agent != cvi:
                return Response({'error': 'You are not authorized to cancel visit'}, status=status.HTTP_403_FORBIDDEN)
        except Visit.DoesNotExist:
            return Response({'error': 'Visit not found'}, status=status.HTTP_400_BAD_REQUEST)
        if visit.status != 'cancelled':
            return Response({'error': 'Visit is not cancelled'}, status=status.HTTP_400_BAD_REQUEST)
        if action=='drop':
            visit.delete()
            return Response({'message': 'Visit dropped successfully'}, status=status.HTTP_200_OK)
        elif action=='reschedule':
            if  data.get('deadline'):    
                deadline=datetime.strptime(data.get('deadline'),'%Y-%m-%d')
            else:
                deadline=cvi.deadline
            other_visits=Visit.objects.filter(agent=cvi,validated=1,status='scheduled',deadline=deadline).order_by('order')
            max_order=0
            for visit in other_visits:
                if visit.order>max_order:
                    max_order=visit.order
            visit.order=max_order+1
            visit.status='scheduled'
            visit.deadline=deadline
            visit.cancel_proof=None
            print(f"new visit order: {visit.order}")
            print(f"new deadline: {visit.deadline}")
            visit.save()
            return Response({'message': 'Visit rescheduled successfully'}, status=status.HTTP_200_OK)
