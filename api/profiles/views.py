from rest_framework.generics import (
    RetrieveUpdateAPIView,
    ListCreateAPIView,
)
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny
from .serializers import ProfileSerializer
from api.permissions import IsOwnerOfProfile


class UserListCreateAPIView(ListCreateAPIView):
    """This view allows to get a list of all users or
    to create a new one.

    Username requirements:
    Your username must be unique.

    Password requirements:
    Your password must contain at least 8 characters.
    """
    queryset = User.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [AllowAny]


class UserDetailAPIView(RetrieveUpdateAPIView):
    """This view allows to:

    * retrieve user`s profile
    * update profile if you are the owner
    * update profile if you have staff status
    """

    queryset = User.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [IsOwnerOfProfile]
