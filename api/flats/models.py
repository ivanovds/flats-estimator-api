from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator


class Flat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    floor = models.PositiveIntegerField(validators=[MinValueValidator(1), MaxValueValidator(48)])
    floors_count = models.PositiveIntegerField(validators=[MinValueValidator(4), MaxValueValidator(48)])
    total_square_meters = models.FloatField(validators=[MinValueValidator(20), MaxValueValidator(200)])
    latitude = models.FloatField(validators=[MinValueValidator(50), MaxValueValidator(51)])
    longitude = models.FloatField(validators=[MinValueValidator(30), MaxValueValidator(31)])
    center_distance = models.PositiveIntegerField(null=True, validators=[MaxValueValidator(20000)])
    metro_distance = models.PositiveIntegerField(null=True)
    azimuth = models.PositiveIntegerField(null=True, validators=[MinValueValidator(0), MaxValueValidator(360)])
    predicted_price_metr = models.PositiveIntegerField(null=True)
    predicted_price_total = models.PositiveIntegerField(null=True)
