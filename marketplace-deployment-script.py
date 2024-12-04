import google.cloud.marketplace_v2 as marketplace
from google.cloud import storage
import yaml
import os

class GameDevVMDeployment:
    def __init__(self, project_id, region='us-central1'):
        self.project_id = project_id
        self.region = region
        self.marketplace_client = marketplace.CloudMarketplaceClient()
        self.storage_client = storage.Client()

    def prepare_vm_package(self):
        """
        Przygotowanie pakietu dla Google Cloud Marketplace
        """
        # Wczytanie konfiguracji
        with open('marketplace.yaml', 'r') as file:
            marketplace_config = yaml.safe_load(file)
        
        # Bucket dla plików dystrybucyjnych
        bucket_name = f"{self.project_id}-game-dev-vm"
        bucket = self.storage_client.create_bucket(bucket_name)
        
        # Upload plików konfiguracyjnych
        files_to_upload = [
            'marketplace.yaml',
            'game-dev-vm-setup.sh'
        ]
        
        for filename in files_to_upload:
            blob = bucket.blob(filename)
            blob.upload_from_filename(filename)
        
        return bucket_name

    def submit_solution(self, bucket_name):
        """
        Złożenie rozwiązania do Google Cloud Marketplace
        """
        solution_metadata = {
            'name': 'Game Development AI VM',
            'version': '1.0.0',
            'description': 'Zaawansowana maszyna wirtualna dla deweloperów gier z AI',
            'packagePath': f'gs://{bucket_name}'
        }
        
        try:
            response = self.marketplace_client.submit_solution(
                parent=f'projects/{self.project_id}',
                solution=solution_metadata
            )
            print(f"Rozwiązanie zostało złożone: {response}")
        except Exception as e:
            print(f"Błąd podczas składania rozwiązania: {e}")

def main():
    deployment = GameDevVMDeployment(project_id='twoj-projekt-id')
    bucket_name = deployment.prepare_vm_package()
    deployment.submit_solution(bucket_name)

if __name__ == "__main__":
    main()
