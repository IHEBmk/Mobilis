"""
URL configuration for mobilis_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from server.AgencieView import AddAgence, UploadAgencies
from server.cviViews import CVICoordinatesAPIView, CVIDetailsAPIView, CVILastVisitsAPIView, CVIProfileAPIView, CVIVisitPerformanceAPIView, CVIVisitsRealizedVsGoal
from server.CoordinatesView import GetCoordinates, RefreshCoordinates
from server.PerformanceView import getGlobalPerformancePDV, getVisitPerformance
from server.UserViews import AssignZoneView, GenerateAgents, GenerateManager, GenerateWilayaManagers, GenerateWilayasManagers, GetUsers, LoginView, SignupView
from server.CommuneView import Commune_to_supabase, GetCommunes
from server.PdvViews import DeletePdv,GetOnePdv, GetPdv, Pdv_To_supabase, UpdateStatusPdv
from server.VisitView import ClancelVisit, GetVisitsPlan, MakePlanning, ValidatePlanning, VisitPdv
from server.WilayaView import GetGeojson, GetWilayas, Wilaya_to_supabase
from rest_framework_simplejwt.views import TokenRefreshView

from server.ZoneView import GenerateZones
from server.dashboardViews import AgentCoordinatesAPIView, AverageVisitDuration, CommerciauxActifs, DashboardStats, LastWeekVisits, VisitedPDVPercentage, VisitsRealizedVsGoal, ZoneStatsAPIView, pdvVisited




urlpatterns = [
    
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('assign/', AssignZoneView.as_view(), name='assignzonetouser'),
    path('pdv_to_supa/', Pdv_To_supabase.as_view(), name='pdv_to_supa'),
    path('wilaya_to_supa/', Wilaya_to_supabase.as_view(), name='wilaya_to_supa'),
    path('commune_to_supa/', Commune_to_supabase.as_view(), name='commune_to_supa'),
    path('UploadAgencies/', UploadAgencies.as_view(), name='UploadAgencies'),
    path('GetPdv/', GetPdv.as_view(), name='GetPdv'),
    path('UpdateStatusPdv/', UpdateStatusPdv.as_view(), name='UpdateStatusPdv'),
    path('VisitPdv/', VisitPdv.as_view(), name='VisitPdv'),
    path('MakePlanning/', MakePlanning.as_view(), name='MakePlanning'),
    path('ValidatePlanning/', ValidatePlanning.as_view(), name='ValidatePlanning'),
    path('GetVisitsPlan/', GetVisitsPlan.as_view(), name='GetVisitsPlan'),
    path('ClancelVisit/', ClancelVisit.as_view(), name='ClancelVisit'),
    path('cvi/list/', CVIDetailsAPIView.as_view(), name='cvi_list'),
    path('cvi/<uuid:cvi_id>/profile/', CVIProfileAPIView.as_view(), name='cvi_profile'),
    path('cvi/<uuid:cvi_id>/last_visits/', CVILastVisitsAPIView.as_view(), name='cvi_last_visits'),
    path('cvi/<uuid:cvi_id>/visits_realized_vs_goal/', CVIVisitsRealizedVsGoal.as_view(), name='cvi_visits_realized_vs_goal'),
    path('cvi/<uuid:cvi_id>/visit_performance/', CVIVisitPerformanceAPIView.as_view(), name='cvi_visit_performance'),   
    path('cvi/<uuid:cvi_id>/coordinates/', CVICoordinatesAPIView.as_view(), name='cvi_coordinates'),
    path('dashboardstats/', DashboardStats.as_view(), name='dashboardstats'),
    path('DeletePdv/', DeletePdv.as_view(), name='DeletePdv'),
    path('GetCoordinates/', GetCoordinates.as_view(), name='GetCoordinates'),
    path('RefreshCoordinates/', RefreshCoordinates.as_view(), name='RefreshCoordinates'),
    path('getGlobalPerformancePDV/', getGlobalPerformancePDV.as_view(), name='getGlobalPerformancePDV'),
    path('getVisitPerformance/', getVisitPerformance.as_view(), name='getVisitPerformance'),
    path('GetWilayas/', GetWilayas.as_view(), name='GetWilayas'),
    path('GetCommunes/', GetCommunes.as_view(), name='GetCommunes'),
    path('GenerateWilayasManagers/', GenerateWilayasManagers.as_view(), name='GenerateWilayasManagers'),
    path('GenerateWilayaManagers/', GenerateWilayaManagers.as_view(), name='GenerateWilayaManagers'),
    path('GenerateManager/', GenerateManager.as_view(), name='GenerateManager'),
    path('GenerateAgents/', GenerateAgents.as_view(), name='GenerateAgents'),
    path('GetOnePdv/', GetOnePdv.as_view(), name='GetOnePdv'),
    path('GetUsers/', GetUsers.as_view(), name='GetUsers'),
    path('GetGeojson/', GetGeojson.as_view(), name='GetUsers'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('AddAgence/', AddAgence.as_view(), name='AddAgence'),
    path('GenerateZones/', GenerateZones.as_view(), name='GenerateZones'),
]
