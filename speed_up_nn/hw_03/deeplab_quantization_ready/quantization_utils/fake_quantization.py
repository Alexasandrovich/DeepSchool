from copy import deepcopy
from pathlib import Path
from typing import Tuple
from typing import Union

from torch import nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.quantization.quantize_fx import convert_fx
from torch.quantization.quantize_fx import fuse_fx
from torch.quantization.quantize_fx import prepare_qat_fx
from torch.utils.data import DataLoader


# Этот молодой человек отвественнен за выбор схемы квантования
# Он говорит нам какие узлы Fake квантования нам навесить
# Они также как и обсёрверы просто навешиваются на веса и активации
QCONFIG_MAPPING = get_default_qat_qconfig_mapping("x86")
# В точре по умолчанию два бекенда
# Один для мобилок, а другой для серверов.
# Отличие одно, торчовые квантованные сетки на мобилках не умеют в векторное квантование весов



def fake_quantization(model: nn.Module, data_loader: DataLoader):
    """ Quantize and train network with distillation using quantization aware training

    Parameters
    ----------
    model:
        Model to be quantized
    data_loader:
        Data Loader with Training data.
    log_dir:
        Log dir for checkpoints and tensorboard logs
    q_config_dict:
        quantization config dict
    device:
        device for training

    Returns
    -------
    :
        Quantized and trained model.
    """
    prepared_model = deepcopy(model)
    # Вся магия происходит здесь
    # Torch FX позволяет нам вместо monkey patching
    # Редактировать граф во время рантайма
    # А значит мы можем вставить перед нодой нашу ноду для фейк квантования

    # По умолчанию батчнормы не фьюязтся
    # Но их наличие славно подпортит нам жизнь, потому что из-за них
    # У нас получается сильно несоотвествие между Fake Quant моделью и Quant моделью
    # Есть много способов решить эту проблему, например сделать сильное усреднение для moving average
    # Но я предпочитаю убирать, если это получается
    prepared_model.eval()
    # prepared_model = fuse_fx(prepared_model)
    prepared_model.train()
    model.eval()
    example_input = next(iter(data_loader))
    prepared_model = prepare_qat_fx(
        model=prepared_model,
        qconfig_mapping=QCONFIG_MAPPING,
        example_inputs=(example_input, ),
    )
    return prepared_model
