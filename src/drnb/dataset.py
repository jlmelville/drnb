# create_data_pipeline

#
# preprocess = [
# ("numpyfy", dict(dtype="float32", layout="c"))
# dropNA
# select columns
# "center"
# "zscale"
# ("range_scale", dict(minval=0, maxval=10.0))
# ],
# neighbors = (
#   n_neighbors=[15, 50, 150] # add 1 to be on the safe side?
#   method = "exact",
#   metric = "euclidean",
#   # for method = "exact" or "approximate" we can't know what algo we will get
#   # so need to nest the names inside?
#   method_kwds = dict("annoy"=dict(), hnsw=dict() ... )
# ),
# triplets = (
#   n_triplets_per_point=5,
#   seed=42,
#   metric="l2",
# ), # choose a seed here
# force=True # overwrite if data exists
#
#  data_pipeline.run(data=irisx, target=irisy, name="iris") # override context?
# write into each folder a .pipeline file with the originating pipeline UUID?
# -> return a PipelineResult containing the parameters + actual written paths
# -> also write that to pipelines/<name>.UUID.data.json?

from dataclasses import dataclass, field

from drnb.io.dataset import create_dataset_exporters
from drnb.log import log, log_verbosity


@dataclass
class DatasetPipeline:
    data_sub_dir: str = "data"
    data_exporters: list = field(default_factory=list)
    target_exporters: list = field(default_factory=list)

    def run(self, name, data, target=None, verbose=False):
        with log_verbosity(verbose):
            log.info("Writing data for %s", name)
            for exporter in self.data_exporters:
                exporter.export(name, data, sub_dir=self.data_sub_dir, suffix="data")

            if target is not None and self.target_exporters:
                log.info("Writing target for %s", name)
                for exporter in self.target_exporters:
                    exporter.export(
                        name, target, sub_dir=self.data_sub_dir, suffix="target"
                    )


def create_data_pipeline(
    data_export,
    target_export=None,
    verbose=False,
):
    with log_verbosity(verbose):
        #     embedder = create_embedder(method)
        #     evaluators = create_evaluators(eval_metrics)
        #     plotter = nbplot.create_plotter(plot)
        #     exporters = embedio.create_embed_exporters(get_embedder_name(method), export)
        data_exporters = create_dataset_exporters(data_export)
        target_exporters = create_dataset_exporters(target_export)

        return DatasetPipeline(
            #         embedder=embedder,
            #         evaluators=evaluators,
            #         plotter=plotter,
            data_exporters=data_exporters,
            target_exporters=target_exporters
            #         verbose=verbose,
        )
