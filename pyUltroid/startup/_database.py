# Ultroid - UserBot
# Copyright (C) 2021-2022 TeamUltroid
#
# This file is a part of < https://github.com/TeamUltroid/Ultroid/ >
# PLease read the GNU Affero General Public License in
# <https://github.com/TeamUltroid/pyUltroid/blob/main/LICENSE>.

__all__ = ("LocalDB", "MongoDB", "RedisDB", "SqlDB")

from ast import literal_eval
from os import environ, path, system
from copy import deepcopy
from json import dump, load
from sys import executable

from . import LOGS, HOSTED_ON
from pyUltroid.configs import Var

try:
    from redis.exceptions import ResponseError
except ImportError:

    class ResponseError(AttributeError):
        pass


# ---------------------------------------------------------------------------------------------


class _BaseDatabase:
    __slots__ = ("_cache",)

    def __init__(self, *args, **kwargs):
        if self.to_cache:
            self._cache = {}
            self._re_cache()

    def ping(self):
        return 1

    @property
    def usage(self):
        return 0

    def del_key(self, key):
        if self.to_cache:
            self._cache.pop(key, None)
        return self.delete(key)

    def _get_data(self, key=None, data=None):
        if key:
            try:
                data = self.get(str(key))
            except ResponseError:
                return LOGS.error(f"'WRONGTYPE' Key Error for {key!r}")
            except Exception:
                return LOGS.debug(f"Error getting key {key!r} from DB", exc_info=True)
        if data and type(data) == str:
            try:
                data = literal_eval(data)
            except Exception:
                pass
        return data

    def _re_cache(self, key=None):
        if not self.to_cache:
            raise TypeError("Caching is disabled")
        if key:
            self._cache.pop(key, None)
            return bool(self.get_key(key, force=True))
        for key in self.keys():
            if not key.startswith("__"):
                self.get_key(key, force=True)

    def get_key(self, key, *, force=False):
        if not self.to_cache:
            return self._get_data(key=key)
        elif force:
            # It will sync the cache with db.
            self._cache.pop(key, None)
            value = self._get_data(key=key)
            if not key.startswith("__"):
                self._cache[key] = value
            return value
        return deepcopy(self._cache.get(key))

    def set_key(self, key, value):
        value = self._get_data(data=value)
        if self.to_cache and not key.startswith("__"):
            self._cache[key] = value
        return self.set(str(key), str(value))

    def append(self, key, value):
        if not (data := self.get_key(key)):
            return "Key doesn't exists.."
        value = self._get_data(data=value)
        if type(data) == list:
            data.append(value)
        elif type(data) == dict:
            data = data | value
        elif type(data) == set:
            data.add(value)
        elif type(data) == tuple:
            lst = list(data)
            lst.append(value)
            data = tuple(lst)
        else:
            data = f" {value}"
        return self.set_key(key, data)


# ---------------------------------------------------------------------------------------------


class MongoDB(_BaseDatabase):
    __slots__ = ("dB", "db", "to_cache", "_name")

    def __init__(self, key, to_cache, _name="MongoDB", db_name="UltroidDB"):
        from pymongo import MongoClient

        self.dB = MongoClient(key, serverSelectionTimeoutMS=6500)
        self.db = self.dB[db_name]
        self.to_cache = to_cache
        self._name = _name
        super().__init__()

    def __repr__(self):
        info = f"-cached_keys: {len(self._cache)}" if self.to_cache else ""
        return f"<Ultroid.MongoDB \n\n-name: {self.name} \n{info}\n>"

    @property
    def name(self):
        return self._name

    @property
    def usage(self):
        return self.db.command("dbstats")["dataSize"]

    def ping(self):
        if self.dB.server_info():
            return True

    def keys(self):
        return self.db.list_collection_names()

    def set(self, key, value):
        value = self._get_data(data=value)
        if key in self.keys():
            self.db[key].replace_one({"_id": key}, {"value": str(value)})
        else:
            self.db[key].insert_one({"_id": key, "value": str(value)})
        if self.to_cache:
            self._cache[key] = value
        return True

    def delete(self, key):
        if self.to_cache:
            self._cache.pop(key, None)
        if key in self.keys():
            self.db.drop_collection(key)
            return True

    def get(self, key):
        if x := self.db[key].find_one({"_id": key}):
            return x["value"]


"""
def flushall(self):
        self.dB.drop_database("UltroidDB")
        self._cache.clear()
        return True
"""

# ---------------------------------------------------------------------------------------------

# Thanks to "Akash Pattnaik" / @BLUE-DEVIL1134
# for SQL Implementation in Ultroid.
#
# Please use https://elephantsql.com/ !


class SqlDB(_BaseDatabase):
    __slots__ = ("url", "db", "to_cache", "_name", "_connection", "_cursor")

    def __init__(self, url, to_cache, _name="SQL_DB"):
        self._connection = None
        self._cursor = None
        self.url = url
        self.to_cache = to_cache
        self._name = _name
        try:
            import psycopg2

            self._connection = psycopg2.connect(dsn=url)
            self._connection.autocommit = True
            self._cursor = self._connection.cursor()
            self._cursor.execute(
                "CREATE TABLE IF NOT EXISTS Ultroid (ultroidCli varchar(70))"
            )
        except Exception as error:
            LOGS.critical("Invaid SQL Database!")
            if self._connection:
                self._connection.close()
            quit(0)
        super().__init__()

    @property
    def name(self):
        return self._name

    def __repr__(self):
        info = f"-cached_keys: {len(self._cache)}" if self.to_cache else ""
        return f"<Ultroid.SQLDB \n\n-name: {self.name} \n{info}\n>"

    @property
    def usage(self):
        self._cursor.execute(
            "SELECT pg_size_pretty(pg_relation_size('Ultroid')) AS size"
        )
        data = self._cursor.fetchall()
        return int(data[0][0].split()[0])

    def keys(self):
        self._cursor.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name  = 'ultroid'"
        )  # case sensitive
        data = self._cursor.fetchall()
        return [_[0] for _ in data]

    def get(self, variable):
        try:
            self._cursor.execute(f"SELECT {variable} FROM Ultroid")
        except psycopg2.errors.UndefinedColumn:
            return None
        data = self._cursor.fetchall()
        if not data:
            return None
        if len(data) >= 1:
            for i in data:
                if i[0]:
                    return i[0]

    def set(self, key, value):
        try:
            self._cursor.execute(f"ALTER TABLE Ultroid DROP COLUMN IF EXISTS {key}")
        except (psycopg2.errors.UndefinedColumn, psycopg2.errors.SyntaxError):
            pass
        except Exception as er:
            LOGS.exception(er)
        self._cache[key] = value
        self._cursor.execute(f"ALTER TABLE Ultroid ADD {key} TEXT")
        self._cursor.execute(f"INSERT INTO Ultroid ({key}) values (%s)", (str(value),))
        return True

    def delete(self, key):
        try:
            self._cursor.execute(f"ALTER TABLE Ultroid DROP COLUMN {key}")
        except psycopg2.errors.UndefinedColumn:
            return False
        return True


"""
def flushall(self):
    self._cache.clear()
    self._cursor.execute("DROP TABLE Ultroid")
    self._cursor.execute(
        "CREATE TABLE IF NOT EXISTS Ultroid (ultroidCli varchar(70))"
    )
    return True
"""

# ---------------------------------------------------------------------------------------------


class RedisDB(_BaseDatabase):
    __slots__ = ("db", "to_cache", "_name", "set", "get", "keys", "delete")

    def __init__(
        self,
        host,
        port,
        password,
        to_cache,
        platform="",
        _name="RedisDB",
        logger=LOGS,
        *args,
        **kwargs,
    ):
        if host and ":" in host:
            spli_ = host.split(":")
            host = spli_[0]
            port = int(spli_[-1])
            if host.startswith("http"):
                logger.critical("Your REDIS_URI should not start with http !")
                quit("Redis Error..")
        elif not host or not port:
            logger.critical("Redis error: Host or Port missing..")
            quit("Redis Error..")

        kwargs["host"] = host
        kwargs["password"] = password
        kwargs["port"] = port

        if platform.lower() == "qovery" and not host:
            var, hash_, host, password = "", "", "", ""
            for vars_ in environ:
                if vars_.startswith("QOVERY_REDIS_") and vars.endswith("_HOST"):
                    var = vars_
            if var:
                hash_ = var.split("_", maxsplit=2)[1].split("_")[0]
            if hash:
                kwargs["host"] = environ.get(f"QOVERY_REDIS_{hash_}_HOST")
                kwargs["port"] = environ.get(f"QOVERY_REDIS_{hash_}_PORT")
                kwargs["password"] = environ.get(f"QOVERY_REDIS_{hash_}_PASSWORD")

        from redis import Redis

        self.db = Redis(**kwargs)
        self.set = self.db.set
        self.get = self.db.get
        self.keys = self.db.keys
        self.delete = self.db.delete
        self.to_cache = to_cache
        self._name = _name
        super().__init__()

    @property
    def name(self):
        return self._name

    def ping(self):
        return self.db.ping()

    def __repr__(self):
        info = f"-cached_keys: {len(self._cache)}" if self.to_cache else ""
        return f"<Ultroid.RedisDB \n\n-name: {self._name} \n{info}\n>"

    @property
    def usage(self):
        return sum(self.db.memory_usage(x) for x in self.keys())


"""
def real(self, key):
    try:
        return self.db.get(key)
    except ResponseError:
        try:
            return self.db.lrange(key, 0, -1)
        except ResponseError:
            try:
                return self.db.hgetall(key)
            except Exception as exc:
                return exc
"""

# --------------------------------------------------------------------------------------------- #


# Source: https://github.com/buddhhu/localdb.json
class LocalDB:
    __slots__ = ("_cache", "name", "db", "to_cache")

    def __init__(self):
        self.db = self
        self.to_cache = True
        self._cache = {}  # Why to read file again and again?
        self.name = "localdb.json"
        if self.ping():
            self._re_cache()
        else:
            self._rewrite_db()

    @property
    def _name(self):
        # self.name == self._name
        return self.name

    def ping(self):
        """ping db - check if database file exists"""
        return path.exists(self.name)

    def __repr__(self):
        info = f"-cached_keys: {len(self._cache)}"
        return f"<Ultroid.LocalDB\n\n-filename: {self.name}\n{info}\n>"

    def _re_cache(self):
        """Load JSON database file in cache"""
        if not self.ping():
            raise FileNotFoundError("DB File has been deleted...")
        with open(self.name, "r", encoding="utf-8") as file:
            try:
                data = load(file)
            except Exception as exc:
                return LOGS.exception(f"Error while decoding local database file..")
        for k, v in data.items():
            try:
                v = literal_eval(str(v))
            except Exception:
                pass
            self._cache[k] = v

    def _rewrite_db(self):
        """Save data to database file"""
        to_write = {}
        for k, v in self._cache.copy().items():
            to_write[k] = str(v)
        with open(self.name, "w", encoding="utf-8") as file:
            try:
                return dump(to_write, file, indent=4)
            except Exception as exc:
                LOGS.exception(f"Error while writing to local database file..")

    def keys(self):
        return self._cache.keys()

    def get_key(self, key, force=False):
        """Get the requested key, uses cache before reading database file."""
        return self._cache.get(key)

    def get(self, key, force=False):
        """Copy of self.get_key"""
        return self.get_key(key, force=force)

    def set_key(self, key, value):
        """Set key with given value"""
        self._cache[key] = value
        self._rewrite_db()
        return True

    def set(self, key):
        """Copy of self.set_key"""
        return self.set_key(key)

    def rename(self, key1, key2):
        """Rename a key with different name."""
        if val := self._cache.get(key1):
            self.del_key(key1)
            return self.set_key(key2, val)
        return False

    def del_key(self, key):
        """Delete a key from database."""
        if self._cache.pop(key, None):
            self._rewrite_db()
            return True

    def delete(self, key):
        """Copy of self.del_key"""
        return self.del_key(key)

    @property
    def usage(self):
        """Size of database file."""
        return path.getsize(self.name) if self.ping() else 0


# ---------------------------------------------------------------------------------------------


def _UltroidDB():
    try:
        if Var.REDIS_URI or Var.REDISHOST:
            LOGS.info("Connecting to Redis Database..")
            try:
                from redis import Redis
            except ImportError:
                LOGS.info("Installing 'Redis' for Database..")
                system(f"{executable} -m pip install -q redis[hiredis]")
                from redis import Redis

            return RedisDB(
                host=Var.REDIS_URI or Var.REDISHOST,
                password=Var.REDIS_PASSWORD or Var.REDISPASSWORD,
                port=Var.REDISPORT,
                platform=HOSTED_ON,
                decode_responses=True,
                socket_timeout=6,
                retry_on_timeout=True,
                to_cache=True,
                _name="Redis",
            )
        elif Var.MONGO_URI:
            LOGS.info("Connecting to Mongo Database..")
            if Var.MONGO_DBNAME:
                LOGS.debug(f"Using {Var.MONGO_DBNAME} Cluster!")
            try:
                from pymongo import MongoClient
            except ImportError:
                LOGS.info("Installing 'PyMongo' for Database..")
                system(f"{executable} -m pip install -q pymongo[srv]")
                from pymongo import MongoClient

            cluster = Var.MONGO_DBNAME or "UltroidDB"
            return MongoDB(
                key=Var.MONGO_URI, _name="Mongo", to_cache=True, db_name=cluster
            )
        elif Var.DATABASE_URL:
            LOGS.info("Connecting to SQL Database..")
            try:
                import psycopg2
            except ImportError:
                LOGS.info("Installing 'pyscopg2' for Database.")
                system(f"{executable} -m pip install -q psycopg2-binary")
                import psycopg2

            return SqlDB(url=Var.DATABASE_URL, to_cache=True, _name="SQL")
        else:
            # use Local DB if available.
            LOGS.info("Connecting to Local Database..")
            if not path.exists("localdb.json"):
                LOGS.info("Creating new LocalDB JSON file..")
                with open("localdb.json", "w+") as f:
                    f.write("{}")

            return LocalDB()

            """
            LOGS.critical(
                "No DB requirements fullfilled!\nPlease install Redis, Mongo or SQL dependencies.\n\nTill then using LocalDB as your Database."
            )
            quit(0)
            """
    except BaseException as err:
        LOGS.exception(err)
    quit(0)
