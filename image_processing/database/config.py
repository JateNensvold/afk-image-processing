import jsonpickle


class Config:
    """The "Config" object. Internally based on ``json``."""

    def __init__(self, name: str):
        self.name = name
        self.load_from_file()

    def load_from_file(self):
        """
        Load config from file
        """
        try:
            with open(self.name, 'r', encoding="utf-8") as config_file:
                self._db = jsonpickle.decode(config_file.read())
        except FileNotFoundError:
            self._db = {}

    async def load(self):
        """
        Load config from disk using async loop
        """
        self.load_from_file()

    def _dump(self):
        """
        Dump config file to disk while in use
        """
        with open(self.name, 'w', encoding='utf-8') as tmp:
            tmp.write(jsonpickle.encode(self._db.copy()))

    def save(self):
        """
        Save config to disk using async loop
        """
        self._dump()

    def get(self, key, *args):
        """Retrieves a config entry."""
        return self._db.get(str(key), *args)

    def put(self, key, value):
        """Edits a config entry."""
        self._db[str(key)] = value

    async def remove(self, key):
        """Removes a config entry."""
        del self._db[str(key)]

    def __contains__(self, item):
        return str(item) in self._db

    def __getitem__(self, item):
        return self._db[str(item)]

    def __len__(self):
        return len(self._db)

    def all(self):
        """Returns entire config dictionary"""
        return self._db
