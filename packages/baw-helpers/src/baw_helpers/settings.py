class BawSettings:
    def __init__(self):
        self.domains = {             
                # URLs for different workbenches that exist, with their short names as keys
                "ecosounds": {
                    "api": "https://api.ecosounds.org",
                    "website": "https://www.ecosounds.org"
                },
                "a2o": {
                    "api": "https://api.acousticobservatory.org",
                    "website": "https://data.acousticobservatory.org"
                }
            }
        
        self.default_workbench = "ecosounds"
        

    def update(self, data):
        if "workbench_domains" in data:
            self.domains.update(data["workbench_domains"])
        if "default_workbench" in data:
            if data["default_workbench"] not in self.domains:
                raise ValueError(f"Invalid default_workbench: {data['default_workbench']}. Must be one of: {list(self.domains.keys())}")
            self.default_workbench = data["default_workbench"]

    @property
    def apis(self):
        return [
            domain_info["api"]
            for domain_info in self.domains.values()
            if isinstance(domain_info, dict) and "api" in domain_info
        ]
    
    @property
    def default_api(self):
        return self.domains[self.default_workbench]["api"]
    
    @property
    def default_website(self):
        return self.domains[self.default_workbench]["website"]
    

baw_settings = BawSettings()