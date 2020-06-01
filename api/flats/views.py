from rest_framework.generics import (
    RetrieveUpdateAPIView,
    ListCreateAPIView,
)
from .models import Flat
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import FlatSerializer
from api.permissions import IsOwnerOfFlat
from . import services


class FlatListCreateAPIView(ListCreateAPIView):
    """This view allows to:

    """

    serializer_class = FlatSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            queryset = Flat.objects.all()
        else:
            queryset = Flat.objects.filter(user=self.request.user)

        return queryset

    def perform_create(self, serializer):
        create_update(self, serializer)


class FlatDetailAPIView(RetrieveUpdateAPIView):
    """This view allows to:

    """
    queryset = Flat.objects.all()
    serializer_class = FlatSerializer
    permission_classes = [IsOwnerOfFlat]

    def perform_update(self, serializer):
        create_update(self, serializer)


def create_update(self, serializer):
    latitude = serializer.validated_data['latitude']
    longitude = serializer.validated_data['longitude']
    total_square_meters = serializer.validated_data['total_square_meters']
    center_distance = services.get_center_distance(latitude, longitude)
    metro_distance = services.get_metro_distance(latitude, longitude)
    azimuth = services.get_azimuth(latitude, longitude)
    data = {
        'floor': serializer.validated_data['floor'],
        'floors_count': serializer.validated_data['floors_count'],
        'total_square_meters': total_square_meters,
        'latitude': latitude,
        'longitude': longitude,
        'center_distance': center_distance,
        'metro_distance': metro_distance,
        'azimuth': azimuth,
    }
    predicted_price_total = services.predict_price(data)
    predicted_price_metr = int(predicted_price_total/total_square_meters)

    serializer.save(
        user=self.request.user,
        center_distance=center_distance,
        metro_distance=metro_distance,
        azimuth=azimuth,
        predicted_price_total=predicted_price_total,
        predicted_price_metr=predicted_price_metr,
    )