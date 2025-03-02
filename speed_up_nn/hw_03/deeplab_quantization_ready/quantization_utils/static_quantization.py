from copy import deepcopy
from itertools import islice

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import convert_fx
from torch.quantization.quantize_fx import prepare_fx
from torch.utils.data import DataLoader

# Этот молодой человек отвественнен за выбор схемы квантования
# Он говорит нам о том какие ОБСЁРВЕРЫ повесить на веса и активации
# Обсёрверы в торче, это просто модули которые навешиваются на веса и активации
# И во время forward pass собирают статистики для подбора порогов
QCONFIG_MAPPING = get_default_qconfig_mapping("x86")
# В точре по умолчанию два бекенда
# Один для мобилок, а другой для серверов.
# Отличие одно, торчовые квантованные сетки на мобилках не умеют в векторное квантование весов


def quantize_static(
        model: nn.Module,
        data_loader: DataLoader,
        num_batches: int = 8,
        device: str = 'cuda:0'
) -> nn.Module:
    """ Quantize model statically using min max calibration

    Parameters
    ----------
    model:
        Model to be quantized.
    data_loader:
        Data loader with calibration data. May be validation dataset.
    num_batches:
        Number of batches for calibration.
    q_config_dict:
        quantization config dict
    device:
        device for calibration

    Returns
    -------
    :
        Quantized and trained model.
    """
    prepared_model = deepcopy(model)
    # Вся магия происходит здесь
    # Torch FX позволяет нам вместо monkey patching
    # Редактировать граф во время рантайма
    # А значит мы можем вставить перед нодой нашу ноду для калибровки
    prepared_model.eval()
    example_input = next(iter(data_loader))
    prepared_model = prepare_fx(
        model=prepared_model,
        qconfig_mapping=QCONFIG_MAPPING,
        example_inputs=(example_input, ),
    )
    # print(prepared_model.print_readable())
    # До FX все подобные трюки использовали monkey patching
    # И Тайные знания о структуре модели, некоторые писали свои трейсеры
    # Потому что нормально квантовать, не зная как связаные между собой слои, не получится.
    # Например пайторчи предлагали явно самим встраивать узлы квантования и деквантования в начале и в конце сети
    # Потому что автоматически понять что это нужно сделать нельзя было.

    device = torch.device(device)

    prepared_model.eval()
    prepared_model.to(device)

    with torch.no_grad():
        # Для калиброваки сильно много данных не надо.
        for image, _ in islice(data_loader, num_batches):
            prepared_model(image.to(device))

    # Собственной в данной строчке и происходит ускорение и квантование
    # Моделька до этого была обычной, просто собирала необходимые статистики
    # А вот теперь мы заменяем операции на квантованные в int8
    prepared_model.cpu()
    quantized_model = convert_fx(prepared_model)
    return quantized_model
