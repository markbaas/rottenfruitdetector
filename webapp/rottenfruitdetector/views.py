from django.shortcuts import render
from rottenfruitdetector.classifier import classify, load_image
import requests
from io import BytesIO


def index(request):
    """View function for home page of site."""
    # Render the HTML template index.html with the data in the context variable
    category = None
    is_rotten = 0
    if request.method == 'POST':
        if request.POST['url']:
            fd = BytesIO(requests.get(request.POST['url']).content)
        elif 'file' in request.FILES:
            fd = request.FILES['file']

        x = load_image(fd)

        prediction = classify(x, 'fruits')
        try:
            category = [x for x, y in zip(['apples', 'banana', 'oranges'], prediction) if y > 0.99][0]
        except IndexError:
            category = 'unknown'

        if category != 'unknown':
            is_rotten = classify(x, category)

        print(category, is_rotten)
    return render(request, 'index.html', context={'category': category, 'is_rotten': is_rotten})

