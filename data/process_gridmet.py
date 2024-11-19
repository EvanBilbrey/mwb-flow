from data.thredds import GridMet
from data.thredds import BBox

bnds = BBox(-108.555, -108, 45.5, 45)

gd = GridMet('pr', start='2020-01-01', end='2020-12-31', bbox=bnds)

t = gd.subset_nc(return_array=True)
t.resample(time='M').mean()
t2d = t.precipitation_amount.isel(time=150)
