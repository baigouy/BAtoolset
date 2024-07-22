import traceback
import gc
import zarr

# from batoolset.img import Img
# pb the store should also store metadata  -−> see how I can do that

__DEBUG__ = True

class ImageCentralStoreZarr:
    store = None

    @classmethod
    def init(cls, folder_path=None):
        try:
            if folder_path is None:
                # Create a Zarr store in memory
                zarr_store = zarr.MemoryStore()
            else:
                zarr_store = zarr.DirectoryStore(folder_path)  # just a single line is sufficient to handle the stuff
            zarr_root = zarr.group(store=zarr_store)
            cls.store = zarr_root
        except:
            traceback.print_exc()
            print('Error zarr store could not be created')

    @classmethod
    def set_store(cls, store):
        if __DEBUG__:
            print('setting store', cls.store)
        # Set the central store (zarr store)
        if isinstance(store, str):
            cls.store = zarr.open(store, mode='a')
        else:
            cls.store = store

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
                cls.store[image_id] = data
            else:
                try:
                    if __DEBUG__:
                        print('deleting image store entry because None')
                    del cls.store[image_id]
                except:
                    pass
            if __DEBUG__:
                print(f"Image {image_id} added to central store.")

            if hasattr(data, 'metadata'):
                cls.store.attrs[image_id] = data.metadata
            else:
                cls.store.attrs[image_id] = None  # just in case there was already a metadata attribute --> make sure to have this
        # else:
        #     if __DEBUG__:
        #         print("Central store is not set.")

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
                img = cls.store[image_id][:]
                metadata = cls.store.attrs.get(image_id, None)
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
        del img

        img = cls.store[image_id] # then we reload it from the store in order not to duplicate it # MEGA TODO -−> check
        metadata = cls.store.attrs.get(image_id, None)

        return metadata, img