import os.path
import traceback
import gc
import zarr

# from batoolset.img import Img
# pb the store should also store metadata  -−> see how I can do that

__DEBUG__ = False

class ImageCentralStore:
    store = None

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
    def load(cls, image_id):
        if False:
            # shall I do gc
            # Check if central store is available and if the image exists in the store
            # gc.collect()
            # Print the number of objects that were garbage collected
            print(f"{gc.collect():,} objects were garbage collected")
        if cls.store is not None:
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
            for file in keys:
                if os.path.exists(file) and os.path.isfile(file):
                    pass
                else:
                    del cls.store[file]
    @classmethod
    def clean(cls, sels):
        cls.clean_non_existing()
        keys = cls.keys()
        if keys:
            # delete entries in cls.store whose keys are not in sels
            keys_to_delete = set(keys) - set(sels)
            for key in keys_to_delete:
                if __DEBUG__:
                    print('deleting entry from the store', key)
                del cls.store[key]

        # # add entries in sels that are not in cls.store
        # keys_to_add = set(sels) - set(cls.store.keys())
        # for key in keys_to_add:
        #     cls.store[key] = None  # or some other default value





