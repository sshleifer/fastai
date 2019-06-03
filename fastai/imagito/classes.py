class ClassUtils:
    imagenette = [
        'n01440764',
        'n02102040',
        'n02979186',
        'n03000684',
        'n03028079',
        'n03394916',
        'n03417042',
        'n03425413',
        'n03445777',
        'n03888257',
    ]

    # add later
    imagewoof = ['n02089973',
                 'n02086240',
                 'n02105641',
                 'n02087394',
                 'n02099601',
                 'n02115641',
                 'n02093754',
                 'n02111889',
                 'n02096294',
                 'n02088364'
                 ]

    @staticmethod
    def from_indices(indices, woof=False):
        classes = []
        for i in indices:
            if woof:
                classes.append(ClassUtils.imagewoof[i])
            else:
                classes.append(ClassUtils.imagenette[i])
        return classes

    @staticmethod
    def filter_classes(image_list, classes=None, woof=False):
        if classes is None:
            return image_list

        class_names = ClassUtils.from_indices(classes, woof=woof)

        def class_filter(path):
            for class_name in class_names:
                if class_name in str(path):
                    return True
            return False

        return image_list.filter_by_func(class_filter)
