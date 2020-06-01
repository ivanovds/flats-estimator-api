from django.contrib.auth.models import User
from rest_framework.serializers import (
    ModelSerializer,
    CharField,
    ValidationError,
)


class ProfileSerializer(ModelSerializer):
    confirm_password = CharField(min_length=8, max_length=100, write_only=True,
                                 style={'input_type': 'password'})

    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'password',
            'confirm_password',
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'confirm_password': {'write_only': True}
        }

    def validate_confirm_password(self, value):
        data = self.get_initial()
        password = data.get("password")
        confirm_password = value
        if password != confirm_password:
            raise ValidationError("Passwords must match.")

    def create(self, validated_data):
        username = validated_data['username']
        password = validated_data['password']
        user_obj = User(username=username, password=password)
        user_obj.set_password(password)
        user_obj.save()
        return user_obj

    def update(self, instance, validated_data):
        instance.username = validated_data.get('username', instance.username)
        instance.password = validated_data.get('first_name', instance.password)
        instance.save()

        return instance
