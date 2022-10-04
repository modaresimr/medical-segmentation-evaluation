from ..common import Object
class MetricABS(Object):

    def __init__(self, num_classes, debug={}):
        self.num_classes = num_classes
        self.debug = debug
        pass

    def set_reference(self, reference, spacing=None):
        self.reference = reference
        self.spacing = spacing if not (spacing is None) else [1, 1, 1]
        self.helper = self.calculate_info(reference, self.spacing, self.num_classes)

    def calculate_info(cls, reference, spacing=None, num_classes=2, **kwargs):
        pass

    def evaluate_single(self, reference, test, spacing=None):
        self.set_reference(reference, spacing)
        return self.evaluate(test)

    def evaluate(self, test):
        pass
