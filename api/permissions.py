from rest_framework.permissions import BasePermission


class IsOwnerOfFlat(BasePermission):
    message = 'You must be the owner of this object.'

    def has_object_permission(self, request, view, obj):

        return obj.user == request.user or request.user.is_staff


class IsOwnerOfProfile(BasePermission):
    message = 'You must be the owner of this object.'

    def has_object_permission(self, request, view, obj):

        return obj == request.user or request.user.is_staff
