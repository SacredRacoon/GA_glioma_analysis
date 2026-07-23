import yaml
from pathlib import Path

class Config:
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфига неть {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_ga_params(self):
        return self.config.get('ga', {})
    
    def get_model_params(self):
        return self.config.get('model', {})
    
    def get_cv_params(self):
        return self.config.get('cv', {})
    
    def get_paths(self):
        return self.config.get('paths', {})
    
    def get_genes(self):
        return self.config.get('genes', [])
    
    def get_test_patient(self):
        return self.config.get('test_patient', {})
    
    def create_output_dirs(self):
        output_dir = Path(self.get_paths().get('output_dir', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def print_config(self):

        print(f"\nGA params")
        for key, value in self.get_ga_params().items():
            print(f"  {key}: {value}")
        
        print(f"\nmodel params")    
        for key, value in self.get_model_params().items():
            print(f"  {key}: {value}")
        
        print(f"\nCV params")    
        for key, value in self.get_cv_params().items():
            print(f"  {key}: {value}")
        
        print(f"\npaths")    
        for key, value in self.get_paths().items():
            print(f"  {key}: {value}")
        
        print(f"\ngenes {len(self.get_genes())}")
        print(f"test mutations {len(self.get_test_patient())}")
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __contains__(self, key):
        return key in self.config