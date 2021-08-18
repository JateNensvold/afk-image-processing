import image_processing.afk.roster.DimensionsObject as DO


class RowItem():
    def __str__(self):
        return "({} {})".format(self.name, self.dimensions)

    def __init__(self, dimensions: tuple, name: str = None):
        """
        Create RowItem object from dimensions and name

        Args:
            dimensions: tuple containing (x,y, width, height)
            name: name of RowItem
        """
        self.dimensions = DO.DimensionsObject(dimensions)
        if name is None:
            self.name = id(self)
        else:
            self.name = name
        self.alias = set()

        self.alias.add(id(self))
        self.column = None

    def __getitem__(self, index: int) -> int:
        """
        Return dimension attribute found at 'index'

        Return:
            x on 0
            y on 1
            w on 2
            h on 3
        Raise:
            indexError on other
        """
        return self.dimensions[index]

    def merge(self, row_item: "RowItem", **dimension_object_kwargs):
        """
        Combine two RowItem into the existing RowItem object
        Args:
            row_item: RowItem to absorb
        """
        self.dimensions.merge(row_item.dimensions, **dimension_object_kwargs)
        self.alias.add(row_item.name)

        self.alias.update(row_item.alias)
        self.alias.remove(id(row_item))
