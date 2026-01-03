from django.db import models
import json

class UploadedImage(models.Model):
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        if self.image:
            return self.image.name
        return f"Image uploaded at {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"
    
class Chatbot(models.Model):
    user_input = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Enhanced fields for RAG responses
    sources = models.TextField(blank=True, null=True)  # JSON list of source documents
    images = models.TextField(blank=True, null=True)   # JSON list of image URLs
    web_info = models.TextField(blank=True, null=True) # Web search results if used
    documents_found = models.IntegerField(default=0)
    
    def __str__(self):
        return f"User Input: {self.user_input[:50]}"
    
    def get_sources_list(self):
        """Return sources as a Python list."""
        if self.sources:
            try:
                return json.loads(self.sources)
            except json.JSONDecodeError:
                return []
        return []
    
    def get_images_list(self):
        """Return images as a Python list of dicts."""
        if self.images:
            try:
                return json.loads(self.images)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_sources(self, sources_list):
        """Set sources from a Python list."""
        self.sources = json.dumps(sources_list)
    
    def set_images(self, images_list):
        """Set images from a Python list."""
        self.images = json.dumps(images_list)