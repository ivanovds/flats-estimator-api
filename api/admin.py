"""Admin interface"""

from django.contrib import admin
from api.flats.models import Flat


class FlatModelAdmin(admin.ModelAdmin):
    list_display = ["user", "floor", "floors_count",
                    "total_square_meters", "center_distance",
                    "center_distance", "predicted_price_total"]

    search_fields = ["user"]

    class Meta:
        model = Flat


admin.site.register(Flat, FlatModelAdmin)

