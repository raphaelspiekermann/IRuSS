import rootutils
import torch  # noqa

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from iruss.models.backbones import get_model, list_models

print(*list_models(), sep="\n")

rn = get_model("resnet18_im1k")
print(rn.describe())

exit()

for model_name in list_models():
    if model_name.endswith("no_pt"):
        model = get_model(model_name)
        print(model.describe())
        print()


exit()

model = get_model("convnext_tiny_pt_only")
print(model._probe_in_channels())
print(model._probe_out_channels())
print(model._probe_total_stride())

exit()
# print(_ConvNext_Weights)

for b in _BUILDERS:
    model = _BUILDERS[b]()

    # print(model)

    break
