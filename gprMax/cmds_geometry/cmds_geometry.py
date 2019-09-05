
class UserObjectGeometry:
    """Specific Geometry object."""

    def __init__(self, **kwargs):
        """Constructor."""
        self.kwargs = kwargs
        # define the order of priority for calling create()
        self.order = None
        # hash command
        self.hash = '#example'

        # auto translate
        self.autotranslate = True

    def __str__(self):
        """Readble user string as per hash commands."""
        s = ''
        for k, v in self.kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                v = ' '.join([str(el) for el in v])
            s += str(v) + ' '

        return '{}: {}'.format(self.hash, s[:-1])

    def params_str(self):
        """Readble string of parameters given to object."""
        return self.hash + ': ' + str(self.kwargs)

    def create(self, grid, uip):
        """Create the object and add it to the grid."""
