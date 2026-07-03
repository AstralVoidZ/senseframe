"""冒烟测试：验证顶层 import 和核心 API 可调用。

import 失败直接抛异常——冒烟测试就是要发现 import 问题。
"""


def test_import_senseframe():
    import senseframe
    assert senseframe is not None


def test_version():
    import senseframe
    assert hasattr(senseframe, "__version__")
    assert isinstance(senseframe.__version__, str)


def test_exploration_api():
    from senseframe import ExplorationTracker, SearchSpaceMap
    tracker = ExplorationTracker()
    assert tracker is not None


def test_validators_api():
    from senseframe import (
        shape_validator,
        numerical_stability_validator,
        performance_validator,
        transform_pipeline_validator,
    )
    assert shape_validator is not None
    assert numerical_stability_validator is not None
    assert performance_validator is not None
    assert transform_pipeline_validator is not None


def test_skills_api():
    from senseframe import (
        Skill,
        SkillLibrary,
        save_skill,
        load_skill,
        search_skills,
        list_skills,
    )
    assert Skill is not None
    assert SkillLibrary is not None
    assert callable(save_skill)
    assert callable(load_skill)
    assert callable(search_skills)
    assert callable(list_skills)


def test_load_extension_api():
    from senseframe import load_extension
    assert callable(load_extension)


def test_catalog_api():
    from senseframe.scenes.wifi_csi.catalog import (
        list_techniques,
        suggest_pipeline,
        suggest_augment,
    )
    assert callable(list_techniques)
    assert callable(suggest_pipeline)
    assert callable(suggest_augment)
    assert len(list_techniques()) == 13


def test_transforms_api():
    from senseframe.scenes.wifi_csi.transforms import (
        compose_transforms,
        list_transforms,
    )
    assert callable(compose_transforms)
    assert callable(list_transforms)
    assert len(list_transforms()) == 13
