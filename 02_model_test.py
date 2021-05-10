
import fastbook

fastbook.setup_book()

from fastbook import *

learn_path = Path()
# print(learn_path.ls(file_exts='.pkl'))
learn_inf = load_learner(learn_path/'export.pkl')

print(learn_inf)