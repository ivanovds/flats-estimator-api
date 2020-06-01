from rest_framework.decorators import (
    api_view,
    permission_classes,
)
from rest_framework.response import Response
from rest_framework.reverse import reverse
from rest_framework.permissions import AllowAny


@api_view(['GET'])
@permission_classes([AllowAny])
def api_root(request, format=None):
    """Using this api you can:

    * Create account
    * Log in if you already have an account
    * Evaluate your flat
    """
    return Response({
        'users': reverse('user-list-api', request=request, format=format),
        'flats': reverse('flat-list-api', request=request, format=format),
    })
