def test_import():
    import itac_ad
    from itac_ad.models.itac_ad import ITAC_AD
    from itac_ad.components.itr_encoder import VariateTokenEncoder
    assert ITAC_AD and VariateTokenEncoder
