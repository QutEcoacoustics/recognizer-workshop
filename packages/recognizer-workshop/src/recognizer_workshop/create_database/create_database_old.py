# Old create database from downloaded zip of embeddings
# no longer works


    def create_database(self, embeddings_files, overwrite=False):
        """
        Checks if a database already exists at the specified path, and if not, creates one from the files at the location
        specified. If it does allows the user to delete and then creates the database. 
        """

        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        def create_db():

            to_delete = [
              db_path,
              (db_path.parent / Path(db_path.name + "-shm")),
              (db_path.parent / Path(db_path.name + "-wal"))
            ]

            for f in to_delete:
                if f.exists():
                    f.unlink()
            
            if (Path(embeddings_files) / 'filelist.json').exists():
                parquet_filepaths = [Path(embeddings_files) / 'embeddings' / Path(str(f['site_id'])) / Path(str(f['id'])) / 'embeddings.parquet' 
                                     for f in json.load(open(Path(embeddings_files) / 'filelist.json'))]
            else:
                parquet_filepaths = None

            if self.config.max_embeddings_count:
                max_count = self.config.max_embeddings_count
            else:
                max_count = -1

            if not self.config.search_dataset_name:
                print('search_dataset_name not set in config. Please set it before creating the database.')
                return

            print(f'creating db at {db_path.resolve()}')
            db = convert_parquet(parquet_folder = Path(embeddings_files),
                                parquet_filepaths = parquet_filepaths, 
                                db_type = "sqlite_usearch", 
                                dataset_name = self.config.search_dataset_name, 
                                max_count=max_count, 
                                db_path=db_path)
            print(f'db created at {db_path.resolve()} with {db.count_embeddings()} embeddings.')

        if db_path.exists() and not overwrite:

            print(f"DB path already exists at {db_path.resolve()}. (Initialize the DB to see details)")

            button = widgets.Button(description="Delete Existing Database and create it again?",
                                    layout=widgets.Layout(width='auto'))
            def on_button_click(b):
                create_db()
            button.on_click(on_button_click)
            display(button)

        else:
            create_db()
