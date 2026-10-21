# pyre-strict
import logging
import typing as t
from typing import Callable, cast, Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from opensfm import context, pymap
from opensfm.dataset_base import DataSetBase
from opensfm.pymap import TracksManager
from scipy import sparse
from scipy.sparse import csgraph


logger: logging.Logger = logging.getLogger(__name__)


def load_features(
    dataset: DataSetBase, images: List[str]
) -> Tuple[
    Dict[str, NDArray[np.float64]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.int32]],
    Dict[str, NDArray[np.float64]],
]:
    logging.info("Reading features")

    def load_one(im):
        features_data = dataset.load_features(im)
        if not features_data:
            return im, None, None, None, None, None
        features = features_data.points[:, :3]
        colors = features_data.colors
        segmentations = None
        instances = None
        if features_data.semantic:
            segmentations = features_data.semantic.segmentation
            if features_data.semantic.has_instances():
                instances = features_data.semantic.instances
        depths = features_data.depths if features_data.depths is not None else None
        return im, features, colors, segmentations, instances, depths

    # Use context.parallel_map for parallel loading
    results = context.parallel_map(
        load_one, images, dataset.config.get("processes", 1))

    features = {}
    colors = {}
    segmentations = {}
    instances = {}
    depths = {}
    for im, feat, color, seg, inst, depth in results:
        if feat is not None:
            features[im] = feat
            colors[im] = color
            if seg is not None:
                segmentations[im] = seg
            if inst is not None:
                instances[im] = inst
            if depth is not None:
                depths[im] = depth

    return features, colors, segmentations, instances, depths


def load_matches(
    dataset: DataSetBase, images: List[str]
) -> t.Iterator[Tuple[Tuple[str, str], List[Tuple[int, int]]]]:
    """Yield matches for each image pair"""
    for im1 in images:
        try:
            im1_matches = dataset.load_matches(im1)
        except IOError:
            continue
        for im2 in im1_matches:
            if im2 in images:
                yield (im1, im2), im1_matches[im2]


def _link_matches(
    matches: t.Iterator[Tuple[Tuple[str, str], List[Tuple[int, int]]]],
) -> List[List[Tuple[str, int]]]:
    """Group matched features into tracks, all pairs at once.

    The same groups in the same order as a UnionFind fed pair by pair: tracks
    are ordered by the first appearance of any of their features in `matches`
    and list their features in order of first appearance. That order numbers
    the tracks.
    """
    image_ids: Dict[str, int] = {}
    keys = []
    for (im1, im2), pairs in matches:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            continue
        # (image, feature) as one int64, the image id in the high 32 bits
        k = np.empty(2 * len(pairs), dtype=np.int64)
        k[0::2] = (image_ids.setdefault(im1, len(image_ids)) << 32) | pairs[:, 0]
        k[1::2] = (image_ids.setdefault(im2, len(image_ids)) << 32) | pairs[:, 1]
        keys.append(k)
    if not keys:
        return []

    seq = np.concatenate(keys)
    del keys  # the per-pair copies, before np.unique adds its own
    nodes, first_seen, node_of = np.unique(seq, return_index=True, return_inverse=True)
    graph = sparse.coo_matrix(
        (np.ones(len(seq) // 2, dtype=np.int32), (node_of[0::2], node_of[1::2])),
        shape=(len(nodes), len(nodes)),
    )
    _, component = csgraph.connected_components(graph, directed=False)

    # Nodes in first-seen order, grouped by their component's rank: where the
    # component's first-seen node falls in that order.
    order = np.argsort(first_seen)
    comp_in_order = component[order]
    _, comp_first = np.unique(comp_in_order, return_index=True)
    rank = np.empty_like(comp_first)
    rank[np.argsort(comp_first)] = np.arange(len(comp_first))
    grouped = nodes[order[np.argsort(rank[comp_in_order], kind="stable")]]

    names = list(image_ids)
    feats = list(zip([names[i] for i in (grouped >> 32).tolist()],
                     (grouped & 0xFFFFFFFF).tolist()))
    ends = np.cumsum(np.bincount(rank[component])).tolist()
    return [feats[a:b] for a, b in zip([0] + ends[:-1], ends)]


def create_tracks_manager_from_matches_iter(
    features: Dict[str, NDArray[np.float64]],
    colors: Dict[str, NDArray[np.int32]],
    segmentations: Dict[str, NDArray[np.int32]],
    instances: Dict[str, NDArray[np.int32]],
    matches: Callable[[], t.Iterator[Tuple[Tuple[str, str], List[Tuple[int, int]]]]],
    min_length: int,
    depths: Dict[str, NDArray[np.float64]],
    depth_is_radial: bool = True,
    depth_std_deviation: float = 1.0,
) -> TracksManager:
    """Link matches into tracks."""
    logger.debug("Merging features onto tracks")

    logger.debug("Running union-find to find aggregated tracks")
    tracks = [t for t in _link_matches(matches()) if _good_track(t, min_length)]

    logger.debug("Constructing TracksManager from tracks")
    NO_VALUE = pymap.Observation.NO_SEMANTIC_VALUE
    tracks_manager = pymap.TracksManager()
    num_observations = 0
    num_depth_priors = 0
    for track_id, track in enumerate(tracks):
        for image, featureid in track:
            if image not in features:
                continue
            x, y, s = features[image][featureid]
            r, g, b = colors[image][featureid]
            segmentation = (
                int(segmentations[image][featureid])
                if image in segmentations
                else NO_VALUE
            )
            instance = (
                int(instances[image][featureid]
                    ) if image in instances else NO_VALUE
            )

            obs = pymap.Observation(
                x,
                y,
                s,
                int(r),
                int(g),
                int(b),
                featureid,
                segmentation,
                instance,
            )
            tracks_manager.add_observation(image, str(track_id), obs)
            num_observations += 1
            if image in depths:
                depth_value = depths[image][featureid]
                if not np.isnan(depth_value) and not np.isinf(depth_value):
                    std = max(
                        depth_std_deviation * depth_value,  # pyre-ignore
                        depth_std_deviation,
                    )
                    tracks_manager.set_depth_prior(
                        image,
                        str(track_id),
                        pymap.Depth(
                            value=depth_value,  # pyre-ignore
                            std_deviation=std,
                            is_radial=depth_is_radial,
                        ),
                    )
                    num_depth_priors += 1
    logger.info(
        f"{len(tracks)} tracks, {num_observations} observations,"
        f" {num_depth_priors} depth priors added to TracksManager"
    )
    return tracks_manager


def create_tracks_manager(
    features: Dict[str, NDArray[np.float64]],
    colors: Dict[str, NDArray[np.int32]],
    segmentations: Dict[str, NDArray[np.int32]],
    instances: Dict[str, NDArray[np.int32]],
    matches: t.Union[
        Dict[Tuple[str, str], List[Tuple[int, int]]],
        Callable[[], t.Iterator[Tuple[Tuple[str, str], List[Tuple[int, int]]]]]
    ],
    min_length: int,
    depths: Dict[str, NDArray[np.float64]],
    depth_is_radial: bool = True,
    depth_std_deviation: float = 1.0,
) -> TracksManager:
    """
    Link matches into tracks.

    If matches is a dict, it will be wrapped as an iterator.
    If matches is a callable, it will be used directly.
    """
    if callable(matches):
        matches_iter = matches
    else:
        def matches_iter():
            return iter(matches.items())
    return create_tracks_manager_from_matches_iter(
        features,
        colors,
        segmentations,
        instances,
        matches_iter,
        min_length,
        depths,
        depth_is_radial,
        depth_std_deviation,
    )


def common_tracks(
    tracks_manager: pymap.TracksManager, im1: str, im2: str
) -> Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]:
    """List of tracks observed in both images.

    Args:
        tracks_manager: tracks manager
        im1: name of the first image
        im2: name of the second image

    Returns:
        tuple: tracks, feature from first image, feature from second image
    """
    t1 = tracks_manager.get_shot_observations(im1)
    t2 = tracks_manager.get_shot_observations(im2)
    tracks, p1, p2 = [], [], []
    for track, obs in t1.items():
        if track in t2:
            p1.append(obs.point)
            p2.append(t2[track].point)
            tracks.append(track)
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2


TPairTracks = Tuple[List[str], NDArray[np.float64], NDArray[np.float64]]


def all_common_tracks_without_features(
    tracks_manager: pymap.TracksManager,
    min_common: int = 50,
    processes: int = 1,
) -> Dict[Tuple[str, str], List[str]]:
    tracks = all_common_tracks(
        tracks_manager, include_features=False, min_common=min_common, processes=processes
    )
    return cast(Dict[Tuple[str, str], List[str]], tracks)


def all_common_tracks(
    tracks_manager: pymap.TracksManager,
    include_features: bool = True,
    min_common: int = 50,
    processes: int = 1,
) -> Dict[Tuple[str, str], t.Union[TPairTracks, List[str]]]:
    """List of tracks observed by each image pair.

    Args:
        tracks_manager: tracks manager
        include_features: whether to include the features from the images
        min_common: the minimum number of tracks the two images need to have
            in common

    Returns:
        tuple: im1, im2 -> tuple: tracks, features from first image, features
        from second image
    """
    def process_pair(pair):
        im1, im2, size = pair
        if size < min_common:
            return None
        track_ids, points1, points2 = tracks_manager.get_all_common_observations_arrays(
            im1, im2)
        if include_features:
            return (
                (im1, im2),
                (track_ids, points1, points2),
            )
        else:
            return ((im1, im2), track_ids,)

    logger.debug("Computing pairwise connectivity of images")
    pairs = [(k[0], k[1], v)
             for k, v in tracks_manager.get_all_pairs_connectivity().items()]

    logger.debug(f"Gathering pairwise tracks with {processes} processes")
    batch_size = max(1, len(pairs) // (2 * processes))
    results = context.parallel_map(process_pair, pairs, processes, batch_size)

    common_tracks = {}
    for result in results:
        if result is not None:
            k, v = result
            common_tracks[k] = v
    return common_tracks


def _good_track(track: List[Tuple[str, int]], min_length: int) -> bool:
    if len(track) < min_length:
        return False
    images = [f[0] for f in track]
    if len(images) != len(set(images)):
        return False
    return True


def as_weighted_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a weighted graph
    having shots a snodes and weighted by the # of
    common tracks between two nodes.
    """
    images = tracks_manager.get_shot_ids()
    image_graph = nx.Graph()
    for im in images:
        image_graph.add_node(im)
    for k, v in tracks_manager.get_all_pairs_connectivity().items():
        image_graph.add_edge(k[0], k[1], weight=v)
    return image_graph


def as_graph(tracks_manager: pymap.TracksManager) -> nx.Graph:
    """Return the tracks manager as a bipartite graph (legacy)."""
    tracks = tracks_manager.get_track_ids()
    images = tracks_manager.get_shot_ids()

    graph = nx.Graph()
    for track_id in tracks:
        graph.add_node(track_id, bipartite=1)
    for shot_id in images:
        graph.add_node(shot_id, bipartite=0)
    for track_id in tracks:
        for im, obs in tracks_manager.get_track_observations(track_id).items():
            graph.add_edge(
                im,
                track_id,
                feature=obs.point,
                feature_scale=obs.scale,
                feature_id=obs.id,
                feature_color=obs.color,
                feature_segmentation=obs.segmentation,
                feature_instance=obs.instance,
            )
    return graph
