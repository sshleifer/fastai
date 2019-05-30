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

    imagewoof = [
        # TODO
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
    def filter_classes(image_list, classes=None):
        if classes is None:
            return image_list

        class_names = ClassUtils.from_indices(classes)

        def class_filter(path):
            for class_name in class_names:
                if class_name in str(path):
                    return True
            return False

        return image_list.filter_by_func(class_filter)


IMAGENETTE_RENAMER = dict(enumerate(ClassUtils.imagenette))


collection = {}
{prefix for prefix in (foo.split('.')[0] for foo in collection if '.' in foo) if prefix in collection}


{foo.split('.')[0] for foo in collection if '.' in foo and foo.split('.')[0] in collection}
