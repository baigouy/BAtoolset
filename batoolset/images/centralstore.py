import os.path
import gc
import time

# from batoolset.img import Img
# pb the store should also store metadata  -−> see how I can do that

__DEBUG__ = False

def _stamp(image_id):
    """(size, mtime) of the file, None when it is not a readable local file."""
    try:
        st = os.stat(image_id)
        return st.st_size, st.st_mtime_ns
    except (OSError, TypeError, ValueError):
        return None


class ImageCentralStore:
    store = None
    # file version of each entry: a file changed on disk (edited elsewhere,
    # damaged, an online-only cloud file) was served from the store with its
    # old pixels until restart (PyFigures, 2026-11-16)
    _stamps = {}

    @classmethod
    def init(cls):
        cls.store={}

    @classmethod
    def set_store(cls, store):
        if __DEBUG__:
            print('setting store', cls.store)
        if store:
            cls.store=store
        else:
            cls.store = {}

    @classmethod
    def get_store(cls):
        if __DEBUG__:
            print('getting store', cls.store)
        return cls.store

    @classmethod
    def append(cls, image_id, data):
        # Save the image data to the central store
        if cls.store is not None:
            if data is not None:
                metadata = None
                if hasattr(data, 'metadata'):
                    metadata = data.metadata
                cls.store[image_id] = [metadata,data]
                cls._stamps[image_id] = (_stamp(image_id), time.time_ns())
            else:
                try:
                    if __DEBUG__:
                        print('deleting image store entry because None')
                    del cls.store[image_id]
                except:
                    pass
            if __DEBUG__:
                print(f"Image {image_id} added to central store.")

    @classmethod
    def _still_valid(cls, image_id):
        """The stored entry is still the file on disk. File times move in coarse
        steps (about 16 ms on Windows): a file rewritten with the same size
        within one step looks unchanged, so an entry stored less than 2 s
        after the file's last change is never trusted (git's 'racy' rule)."""
        if image_id not in cls._stamps:
            return True                      # stored before stamps existed (set_store)
        (stamp, stored_at) = cls._stamps[image_id]
        now = _stamp(image_id)
        if stamp is None or now != stamp:
            return stamp is None and now is None
        return stored_at - now[1] >= 2_000_000_000

    @classmethod
    def load(cls, image_id):
        if False:
            # shall I do gc
            # Check if central store is available and if the image exists in the store
            # gc.collect()
            # Print the number of objects that were garbage collected
            print(f"{gc.collect():,} objects were garbage collected")
        if cls.store is not None:
            if image_id in cls.store and not cls._still_valid(image_id):
                # the file changed since it was stored: read it again (raises if now unreadable)
                del cls.store[image_id]
                cls._stamps.pop(image_id, None)
            if image_id in cls.store:
                metadata, img = cls.store[image_id]
                if __DEBUG__:
                    print(f"Image {image_id} loaded from central store.")
                return metadata,img
            else:
                metadata, img = cls.reload(image_id)
                return metadata, img

        if __DEBUG__:
            print('No store found --> skipping loading from store for', image_id)
        return None, None

    # maybe put a relaod where the image is necessarily reloaded -−> force an update of the image from source file
    @classmethod
    def reload(cls, image_id):
        from batoolset.img import Img
        if __DEBUG__:
            print('Img called for file', image_id)
        img = Img(image_id,
                  prefer_store_if_available=False)  # pb will this launch an infinite loop -−> maybe so make sure to prevent reload form it
        metadata = None
        if hasattr(img, 'metadata'):
            metadata = img.metadata
        # add image to the store
        cls.append(image_id, img)
        metadata, data = cls.store[image_id]
        return metadata, img

    @classmethod
    def keys(cls):
        if cls.store:
            return cls.store.keys()
        else:
            return None

    @classmethod
    def clean_non_existing(cls):
        keys = cls.keys()
        if keys:
            # iterate over a copy: deleting from the dict while iterating its
            # keys view raises RuntimeError and aborts the cleaning
            for file in list(keys):
                if not (os.path.exists(file) and os.path.isfile(file)):
                    del cls.store[file]

    @classmethod
    def clean(cls, sels):
        # drop the deselected entries FIRST so the (potentially slow, e.g. on
        # disconnected network drives) os.path.exists checks only run on the
        # entries that are actually kept
        keys = cls.keys()
        if keys:
            # delete entries in cls.store whose keys are not in sels
            keys_to_delete = set(keys) - set(sels)
            for key in keys_to_delete:
                if __DEBUG__:
                    print('deleting entry from the store', key)
                del cls.store[key]
        cls.clean_non_existing()

        # # add entries in sels that are not in cls.store
        # keys_to_add = set(sels) - set(cls.store.keys())
        # for key in keys_to_add:
        #     cls.store[key] = None  # or some other default value





