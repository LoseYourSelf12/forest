from typing import Dict, Any
import numpy as np
from core import register_detector
from .yolodtc import YOLODetector, yolo

# Merged YOLO category detectors

class Kat_24_04(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.04 Методы народной медицины",
            stopvec=np.array([1]),
            names=['yolo_yoga_multi'],
            detectors=['yolo_yoga_multi'],
        )

    def __call__(self, local: Dict[str, Any]):
        max_conf = 0.0
        for detector in self.detectors:
            results = yolo(detector, local['img'])
            boxes = results[0].boxes
            if boxes is not None and boxes.conf is not None:
                conf = boxes.conf.cpu().numpy()
                if conf.size > 0:
                    max_conf = max(max_conf, np.max(conf))
        self._vec[0] = max_conf


class Kat_05_07(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.07 QR-код / адрес сайта",
            stopvec=np.array([1, 1]),
            names=['qrcode', 'qrcode_true'],
            detectors=['yolo_qrcode'],
        )


class Kat_21_04(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.04 Безалкогольное пиво/вино",
            stopvec=np.array([1, 0]),
            names=['non_alc_lable', 'alc_bottle'],
            detectors=['yolo_non_alcohol_lable', 'yolo_alc_bottle'],
        )


class Kat_24_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.01 Медицинские услуги",
            stopvec=np.array([1, 0, 1, 1, 1, 1]),
            names=['dentist_items', 'object', 'syringe', 'glove', 'microscope', 'stethoscope'],
            detectors=['yolo_dentist', 'yolo_syringe', 'yolo_med_glove', 'yolo_micro', 'yolo_stethoscope'],
        )


class Kat_24_05(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.05 Методы лечения, профилактики и диагностики",
            stopvec=np.array([1, 1, 1]),
            names=['syringe', 'microscope', 'stethoscope'],
            detectors=['yolo_syringe', 'yolo_micro', 'yolo_stethoscope'],
        )


class Kat_28_06(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.06 Инвест-платформа",
            stopvec=np.array([1, 1, 1, 1]),
            names=['sber_inv', 'vtb_inv', 't_inv', 'alfa_inv'],
            detectors=['yolo_invest'],
        )


class Kat_29_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="29.03 Криптовалюта",
            stopvec=np.array([1, 1, 1, 1, 1]),
            names=['BTC', 'ETH', 'DOGE', 'USDT', 'USDC'],
            detectors=['yolo_crypto_multi'],
        )


class Kat_25_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="25.02 Детское питание",
            stopvec=np.array([1]),
            names=['baby_meal'],
            detectors=['yolo_baby_meal'],
        )


class Kat_05_04(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.04 Запрещенные информ. ресурсы",
            stopvec=np.array([1]),
            names=['banned'],
            detectors=['yolo_restricted_information_resources'],
        )


class Kat_08_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="08.01 Дистанционные продажи",
            stopvec=np.array([1]),
            names=['distance_sales'],
            detectors=['yolo_distance_sales'],
        )


class Kat_28_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.02 Кредит/Ипотека",
            stopvec=np.array([1]),
            names=['credit_or_mortage'],
            detectors=['yolo_credit'],
        )


class Kat_28_11(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.11 Земельные участки",
            stopvec=np.array([0]),
            names=['land_plots'],
            detectors=['yolo_land_plots'],
        )


class Kat_28_12(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.12 Рассрочка",
            stopvec=np.array([1]),
            names=['promo_ind'],
            detectors=['yolo_promo_ind'],
        )


class Kat_10_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="10.01 Социальная реклама",
            stopvec=np.array([1, 1, 1, 1, 1]),
            names=['soc_health', 'soc_simbol', 'soc_project', 'soc_army', 'army_helmet_epaulets'],
            detectors=["yolo_soc_health", "yolo_soc_simbol", "yolo_soc_proj", "yolo_soc_army", "yolo_helmet_and_epaulets"],
        )


class Kat_24_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.02 Медицинские изделия",
            stopvec=np.array([1, 1, 1, 1, 1, 1, 1]),
            names=['glove', 'microscope', 'scalpel', 'mask', 'blood_pressure_monitor', 'stethoscope', 'condom'],
            detectors=["yolo_med_glove", "yolo_micro", "yolo_scalpel", "yolo_mask", "yolo_blood_monit_big", "yolo_stethoscope", "yolo_condoms"],
        )


class Kat_24_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.03 Лекарственные препараты",
            stopvec=np.array([1]),
            names=['pharma'],
            detectors=["yolo_pharma"],
        )


class Kat_24_06(YOLODetector):
    def __init__(self):
        super().__init__(
            name="24.06 МедОрганизация/Аптека",
            stopvec=np.array([1]),
            names=['medicine'],
            detectors=["yolo_medicine"],
        )


class Kat_21_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.01 Алкоголь, демонстрация процесса потребления алкоголя",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors=["yolo_alc_bottle"],
        )


class Kat_21_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.02 Алкомаркет",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors=["yolo_alc_bottle"],
        )


class Kat_21_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="21.03 Бар, ресторан",
            stopvec=np.array([1]),
            names=['alc_bottle'],
            detectors=["yolo_alc_bottle"],
        )


class Kat_59(YOLODetector):
    def __init__(self):
        super().__init__(
            name="59. Табак, табачная продукция, табачные изделия и курительные принадлежности, в том числе трубок, кальянов, сигаретная бумага, зажигалки, демонстрация процесса курения",
            stopvec=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            names=['cigarette', 'pipes', 'smoke', 'smoking', 'no-smoking', 'pack', 'lighter', 'hookah', 'cig_roll'],
            detectors=["yolo_smokers"],
        )


class Kat_49(YOLODetector):
    def __init__(self):
        super().__init__(
            name="49. Оружие и продукция военного назначения",
            stopvec=np.array([1, 1, 1]),
            names=['gun', 'weapon_box', 'edged_weapon'],
            detectors=["yolo_gun", "yolo_weapon_box", "yolo_edged_weapon"],
        )


class Kat_83(YOLODetector):
    def __init__(self):
        super().__init__(
            name="83. Казино (в т.ч. онлайн-казино)",
            stopvec=np.array([1, 1]),
            names=['playing-cards', 'chips'],
            detectors=["yolo_playing_cards", "yolo_chips"],
        )


class Kat_29_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="29.02 Цифровые финансовые активы",
            stopvec=np.array([0, 0, 0, 1]),
            names=['graphics', 'bar', 'plot_bb', 'plot'],
            detectors=['yolo_information_line', 'yolo_information_bar_char', 'yolo_circle'],
        )


class Kat_29_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="29.01 Ценные бумаги",
            stopvec=np.array([0, 1, 0, 1]),
            names=['bar', 'plot_bb', 'graphics', 'plot'],
            detectors=['yolo_information_bar_char', 'yolo_information_line', 'yolo_circle'],
        )


class Kat_27_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="27.01 Спорт + букмекер (основанные на риске игры, пари (азартные игры, букмекерские конторы и т.д.))",
            stopvec=np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
            names=['ball', 'rim', 'ball', '1xBet', 'player', 'referee', 'player_hockey_sticks', 'Boxing_Glove', 'Football', 'basketball_uniform'],
            detectors=['yolo_basket_ball', 'yolo_basket', 'yolo_ball', 'yolo_1xBet', 'yolo_player_and_referee', 'yolo_hockey_stick', 'yolo_gloves', 'yolo_football', 'yolo_bascet_uniform'],
        )


class Kat_05_05(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.05 Физическое лицо",
            stopvec=np.array([1]),
            names=['face'],
            detectors=['yolo_face'],
        )


class Kat_28_14(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.14 Ломбарды",
            stopvec=np.array([0, 1]),
            names=['coin', 'Jewellery'],
            detectors=['yolo_coin', 'yolo_jewelry'],
        )


class Kat_02_01(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.01 Предвыборная Агитация",
            stopvec=np.array([1]),
            names=['party_logo'],
            detectors=['yolo_party_logo'],
        )


class Kat_02_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.02 Иная политическая реклама",
            stopvec=np.array([1]),
            names=['party_logo'],
            detectors=['yolo_party_logo'],
        )


class Kat_02_05(YOLODetector):
    def __init__(self):
        super().__init__(
            name="02.05 Адвокаты",
            stopvec=np.array([1, 1]),
            names=['themis', 'gavel'],
            detectors=['yolo_lawyers'],
        )


@register_detector("yolo_kat_05_03")
class Kat_05_03(YOLODetector):
    def __init__(self):
        super().__init__(
            name="05.03 Информационная продукция",
            stopvec=np.array([1]),
            names=['inform_sign'],
            detectors=['yolo_inform_sign'],
        )


class Kat_09_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="09.02 Иные акции",
            stopvec=np.array([1]),
            names=['promo_ind'],
            detectors=['yolo_promo_ind'],
        )


class Kat_28_07(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.07 Строительство (ДДУ)",
            stopvec=np.array([1]),
            names=['real_estate'],
            detectors=['yolo_real_estate'],
        )


@register_detector("yolo_kat_28_08")
class Kat_28_08(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.08 Застройщик",
            stopvec=np.array([0, 1]),
            names=['real_estate', 'developer_logo'],
            detectors=['yolo_real_estate', 'yolo_developer_logo'],
        )
