# Generated by Django 3.0.6 on 2020-05-31 16:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_auto_20200531_1858'),
    ]

    operations = [
        migrations.RenameField(
            model_name='flat',
            old_name='price_metr',
            new_name='predicted_price_metr',
        ),
        migrations.RenameField(
            model_name='flat',
            old_name='price_total',
            new_name='predicted_price_total',
        ),
        migrations.AddField(
            model_name='flat',
            name='real_price_metr',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='flat',
            name='real_price_total',
            field=models.FloatField(null=True),
        ),
    ]
