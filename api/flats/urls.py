from django.urls import path
from .views import FlatListCreateAPIView, FlatDetailAPIView

urlpatterns = [
    path('', FlatListCreateAPIView.as_view(), name='flat-list-api'),
    path('<int:pk>/', FlatDetailAPIView.as_view(), name='flat-api-detail'),

]
