from typing import List, Dict, Tuple
import decimal

def round_digits(src: float, rounddigit: int = None, method='decimal'):
    """
    指定桁数で小数を丸める
    Parameters
    ----------
    srcdict : Dict[str, float]
        丸め対象のDict
    rounddigit : int
        フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
    method : int
        桁数決定手法（'decimal':小数点以下, 'sig':有効数字(Decimal指定), 'format':formatで有効桁数指定）
    """
    if method == 'decimal':
        return round(src, rounddigit)
    elif method == 'sig':
        with decimal.localcontext() as ctx:
            ctx.prec = rounddigit
            return ctx.create_decimal(src)
    elif method == 'format':
        return '{:.{width}g}'.format(src, width=rounddigit)

def round_dict_digits(srcdict: Dict[str, float], rounddigit: int = None, method='decimal'):
    """
    指定桁数でDictの値を丸める
    Parameters
    ----------
    srcdict : Dict[str, float]
        丸め対象のDict
    rounddigit : int
        フィッティング線の表示範囲（標準偏差の何倍まで表示するか指定）
    method : int
        桁数決定手法（'decimal':小数点以下, 'sig':有効数字(Decimal指定), 'format':formatで有効桁数指定）
    """
    dstdict = {}
    for k, v in srcdict.items():
        if rounddigit is not None and isinstance(v, float):
            dstdict[k] = round_digits(v, rounddigit=rounddigit, method=method)
        else:
            dstdict[k] = v
    return dstdict