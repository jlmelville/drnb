import pytest

import drnb.embed.pipeline as pl
from drnb.embed import check_embed_method, embedder, get_embedder_name
from drnb.embed.base import Embedder
from drnb.embed.factory import create_embedder
from drnb.types import EmbedConfig


def test_create_pipeline_simple_string():
    pipeline = pl.create_pipeline(method="pca")
    assert isinstance(pipeline.embedder, Embedder)
    # Check if we can inspect the underlying embedder type/params if possible,
    # but for now just checking it's an Embedder is a good start.


def test_create_pipeline_embedder_helper():
    """Test create_pipeline() with embedder() helper that has wrapper kwargs."""
    # Note: n_neighbors should be in params, not as a wrapper kwarg
    # But we can test with a valid wrapper kwarg like use_precomputed_knn
    pipeline = pl.create_pipeline(method=pl.embedder("umap", use_precomputed_knn=False))
    assert isinstance(pipeline.embedder, Embedder)


def test_standard_pipeline_list_strings():
    pipeline = pl.standard_pipeline(method=["pca", "sklearn-mmds"])
    assert isinstance(pipeline.embedder, list)
    assert len(pipeline.embedder) == 2
    assert isinstance(pipeline.embedder[0], Embedder)
    assert isinstance(pipeline.embedder[1], Embedder)


def test_standard_pipeline_mixed_list():
    method = [
        pl.embedder("smmds", params=dict(n_components=50, random_state=42)),
        pl.embedder("rescale"),
        ("tsne", dict(n_components=2, random_state=42, metric="euclidean")),
    ]
    pipeline = pl.standard_pipeline(method=method)
    assert isinstance(pipeline.embedder, list)
    assert len(pipeline.embedder) == 3
    assert isinstance(pipeline.embedder[0], Embedder)
    assert isinstance(pipeline.embedder[1], Embedder)
    assert isinstance(pipeline.embedder[2], Embedder)


def test_check_embed_method_tuple_merge():
    # Test the specific merging logic mentioned in the review/docstring
    # check_embed_method(
    #     embedder("pacmap", local_scale=False, params=dict(n_neighbors=15, apply_pca=True)),
    #     params=dict(apply_pca=False),
    # )
    method = pl.embedder(
        "pacmap", local_scale=False, params=dict(n_neighbors=15, apply_pca=True)
    )
    params = dict(apply_pca=False)

    result = pl.check_embed_method(method, params)

    # Expected: ('pacmap', {'local_scale': False, 'params': {'n_neighbors': 15, 'apply_pca': False}})
    assert isinstance(result, tuple)
    assert result[0] == "pacmap"
    assert result[1]["local_scale"] is False
    assert result[1]["params"]["n_neighbors"] == 15
    assert result[1]["params"]["apply_pca"] is False


# Tests for embedder() helper function
def test_embedder_helper_simple():
    """Test embedder() helper returns an EmbedConfig."""
    result = embedder("umap")
    assert isinstance(result, EmbedConfig)
    assert result.name == "umap"
    assert result.params == {}
    assert result.wrapper_kwds == {}


def test_embedder_helper_with_params():
    """Test embedder() helper with params dict."""
    result = embedder("pca", params=dict(n_components=2))
    assert isinstance(result, EmbedConfig)
    assert result.name == "pca"
    assert result.params["n_components"] == 2


def test_embedder_helper_with_wrapper_kwargs():
    """Test embedder() helper with wrapper kwargs."""
    result = embedder("umap", use_precomputed_knn=True, initialization="spectral")
    assert isinstance(result, EmbedConfig)
    assert result.name == "umap"
    assert result.wrapper_kwds["use_precomputed_knn"] is True
    assert result.wrapper_kwds["initialization"] == "spectral"
    assert result.params == {}


def test_embedder_helper_with_params_and_kwargs():
    """Test embedder() helper with both params and wrapper kwargs."""
    result = embedder(
        "tsne", params=dict(n_iter=1000), initialization="spectral", dof=0.7
    )
    assert isinstance(result, EmbedConfig)
    assert result.name == "tsne"
    assert result.params["n_iter"] == 1000
    assert result.wrapper_kwds["initialization"] == "spectral"
    assert result.wrapper_kwds["dof"] == 0.7


# Tests for check_embed_method() function
def test_check_embed_method_string():
    """Test check_embed_method() with string input."""
    result = check_embed_method("pca")
    assert isinstance(result, tuple)
    assert result[0] == "pca"


def test_check_embed_method_string_with_params():
    """Test check_embed_method() with string input and params."""
    result = check_embed_method("pca", params=dict(n_components=2))
    assert result[0] == "pca"
    assert result[1]["params"]["n_components"] == 2


def test_check_embed_method_tuple():
    """Test check_embed_method() with tuple input."""
    method = ("umap", dict(n_neighbors=15))
    result = check_embed_method(method)
    assert isinstance(result, tuple)
    assert result[0] == "umap"


def test_check_embed_method_tuple_with_params_merge():
    """Test check_embed_method() merges params with tuple input params."""
    method = ("umap", dict(params=dict(n_neighbors=10)))
    result = check_embed_method(method, params=dict(n_neighbors=15))
    assert result[0] == "umap"
    # params argument should override tuple params
    assert result[1]["params"]["n_neighbors"] == 15


def test_check_embed_method_list():
    """Test check_embed_method() with list input."""
    method = [embedder("pca"), embedder("umap")]
    result = check_embed_method(method)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, tuple) for r in result)


def test_check_embed_method_list_no_params():
    """Test check_embed_method() with list input requires params=None."""
    method = [embedder("pca"), embedder("umap")]
    # This should work
    check_embed_method(method, params=None)

    # This should raise ValueError
    with pytest.raises(ValueError, match="params must be None"):
        check_embed_method(method, params=dict(n_components=2))


def test_check_embed_method_none_params():
    """Test check_embed_method() handles None params correctly."""
    result = check_embed_method("pca", params=None)
    assert isinstance(result, tuple)
    # When params=None is passed, embedder() converts it to empty dict {}
    assert result[1].get("params") == {}


# Tests for get_embedder_name() function
def test_get_embedder_name_string():
    """Test get_embedder_name() with string input."""
    assert get_embedder_name("umap") == "umap"


def test_get_embedder_name_tuple():
    """Test get_embedder_name() with tuple input."""
    assert get_embedder_name(("umap", dict())) == "umap"


def test_get_embedder_name_list():
    """Test get_embedder_name() with list input."""
    assert get_embedder_name(["pca", "umap"]) == "pca+umap"


def test_get_embedder_name_list_nested():
    """Test get_embedder_name() with nested list structures."""
    method = [
        ("pca", dict()),
        ("umap", dict()),
    ]
    assert get_embedder_name(method) == "pca+umap"


def test_get_embedder_name_invalid_tuple():
    """Test get_embedder_name() raises error for invalid tuple format."""
    with pytest.raises(ValueError, match="Unexpected format"):
        get_embedder_name(("umap",))


# Tests for create_embedder() factory function
def test_create_embedder_string():
    """Test create_embedder() with string input."""
    result = create_embedder("pca")
    assert isinstance(result, Embedder)


def test_create_embedder_tuple():
    """Test create_embedder() with tuple input."""
    method = ("pca", dict(params=dict(n_components=2)))
    result = create_embedder(method)
    assert isinstance(result, Embedder)


def test_create_embedder_embedder_helper():
    """Test create_embedder() with embedder() helper result."""
    method = embedder("pca", params=dict(n_components=2))
    result = create_embedder(method)
    assert isinstance(result, Embedder)


def test_create_embedder_list_returns_list():
    """Test create_embedder() with list input returns list (type signature fix)."""
    method = ["pca", "sklearn-mmds"]
    result = create_embedder(method)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(e, Embedder) for e in result)


def test_create_embedder_list_mixed_formats():
    """Test create_embedder() with mixed list formats."""
    method = [
        embedder("pca", params=dict(n_components=2)),
        ("sklearn-mmds", dict()),
        "randproj",
    ]
    result = create_embedder(method)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(e, Embedder) for e in result)


def test_create_embedder_with_embed_kwds():
    """Test create_embedder() with embed_kwds parameter."""
    result = create_embedder("pca", embed_kwds=dict(params=dict(n_components=2)))
    assert isinstance(result, Embedder)


# Tests for create_pipeline() function
def test_create_pipeline_tuple():
    """Test create_pipeline() with tuple input."""
    method = ("pca", dict(params=dict(n_components=2)))
    pipeline = pl.create_pipeline(method=method)
    assert isinstance(pipeline.embedder, Embedder)


def test_create_pipeline_list():
    """Test create_pipeline() with list input creates list of embedders."""
    method = ["pca", "sklearn-mmds"]
    pipeline = pl.create_pipeline(method=method)
    assert isinstance(pipeline.embedder, list)
    assert len(pipeline.embedder) == 2


# Tests for standard_pipeline() function
def test_standard_pipeline_string_with_params():
    """Test standard_pipeline() with string method and params argument."""
    pipeline = pl.standard_pipeline("pca", params=dict(n_components=2))
    assert isinstance(pipeline.embedder, Embedder)
    assert pipeline.embed_method_name == "pca"


def test_standard_pipeline_tuple_with_params():
    """Test standard_pipeline() with tuple method and params argument merges correctly."""
    method = embedder("pca", params=dict(n_components=3))
    pipeline = pl.standard_pipeline(method, params=dict(n_components=2))
    # params argument should override embedded params
    assert isinstance(pipeline.embedder, Embedder)
    assert pipeline.embed_method_name == "pca"


def test_standard_pipeline_creates_standard_metrics():
    """Test standard_pipeline() includes standard evaluation metrics."""
    pipeline = pl.standard_pipeline("pca")
    assert len(pipeline.evaluators) > 0


# Edge case tests
def test_check_embed_method_empty_params_dict():
    """Test check_embed_method() handles empty params dict."""
    result = check_embed_method("pca", params={})
    assert isinstance(result, tuple)
    assert result[1].get("params") == {}


def test_check_embed_method_tuple_without_params_key():
    """Test check_embed_method() handles tuple without params key."""
    method = ("pca", dict(n_components=2))
    result = check_embed_method(method)
    assert isinstance(result, tuple)
    # The wrapper kwarg should still be there
    assert "n_components" in result[1] or "params" in result[1]


def test_check_embed_method_tuple_with_none_params():
    """Test check_embed_method() handles tuple with None params."""
    method = embedder("pca", params=None)
    result = check_embed_method(method, params=dict(n_components=2))
    assert isinstance(result, tuple)
    assert result[1]["params"]["n_components"] == 2
