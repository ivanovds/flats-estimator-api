from rest_framework.serializers import (
    ModelSerializer,
    ReadOnlyField,
    ValidationError,
    HyperlinkedIdentityField,
)
from .models import Flat
from . import services


class FlatSerializer(ModelSerializer):
    user = ReadOnlyField(source='user.username')
    edit_flat = HyperlinkedIdentityField(view_name='flat-api-detail')

    class Meta:
        model = Flat
        fields = [
            'edit_flat',
            'user',
            'floor',
            'floors_count',
            'total_square_meters',
            'latitude',
            'longitude',
            'center_distance',
            'metro_distance',
            'predicted_price_metr',
            'predicted_price_total',
        ]
        extra_kwargs = {
            'center_distance': {'read_only': True},
            'metro_distance': {'read_only': True},
            'predicted_price_metr': {'read_only': True},
            'predicted_price_total': {'read_only': True},
        }

    def validate(self, attrs):
        """Check that the center distance not longer than 20 km."""
        latitude = attrs['latitude']
        longitude = attrs['longitude']
        center_distance = services.get_center_distance(latitude, longitude)

        if center_distance > 20000:
            raise ValidationError("Flat must be located no further than 20 km from Kyiv center.")

        return attrs


