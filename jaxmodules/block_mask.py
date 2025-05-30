import jax
from jax import numpy as jnp
from jaxtyping import Array, UInt
from jaxmodules.vectorize import array_from_coords, nested_fori_loop, multi_vmap
from einops import rearrange, einsum
from typing import Callable, NamedTuple, Tuple
import equinox as eqx


def extract_block_size(BLOCK_SIZE: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(BLOCK_SIZE, int):
        return BLOCK_SIZE, BLOCK_SIZE
    else:
        return BLOCK_SIZE


class BlockMask(eqx.Module):
    B: int = eqx.field(static=True)
    H: int = eqx.field(static=True)
    Q_LEN: int = eqx.field(static=True)
    KV_LEN: int = eqx.field(static=True)
    Q_BLOCK_SIZE: int = eqx.field(static=True)
    KV_BLOCK_SIZE: int = eqx.field(static=True)
    # these are the number of PARTIAL blocks in each row
    # we name it this way as opposed to something more descriptive like "partial_kv_num_blocks"
    # to maintain consistency with the pytorch implementation
    kv_num_blocks: UInt[Array, "b h num_q_blocks"]
    # for each row, these are the indices of the PARTIAL blocks
    kv_indices: UInt[Array, "b h num_q_blocks max_kv_blocks_in_row"]

    q_num_blocks: UInt[Array, "b h num_kv_blocks"]
    q_indices: UInt[Array, "b h num_kv_blocks max_q_blocks_in_row"]

    # these are the number of FULL (all-ones) blocks in each row
    full_kv_num_blocks: UInt[Array, "b h num_q_blocks"]
    full_kv_indices: UInt[Array, "b h num_q_blocks max_kv_blocks_in_row"]

    full_q_num_blocks: UInt[Array, "b h num_kv_blocks"]
    full_q_indices: UInt[Array, "b h num_kv_blocks max_q_blocks_in_row"]

    # function used to fill in partial blocks
    mask_mod: Callable[[Array, Array], Array] = eqx.field(static=True)

    def get_mask_for_partial_block(self, b, h, q_block_idx, kv_block_idx):
        """
        Get the mask for a partial block at the specified indices.

        Args:
            b: batch index
            h: head index
            q_block_idx: query block index
            kv_block_idx: key-value block index

        Returns:
            A boolean mask of shape (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for the specified block
        """

        def coord_fn(rel_q_idx, rel_k_idx):
            q_idx = rel_q_idx + q_block_idx * self.Q_BLOCK_SIZE
            k_idx = rel_k_idx + kv_block_idx * self.KV_BLOCK_SIZE
            return self.mask_mod(b, h, q_idx, k_idx)

        return array_from_coords(
            shape=(self.Q_BLOCK_SIZE, self.KV_BLOCK_SIZE),
            fn=coord_fn,
        )

    def materialize_mask(self):
        """
        Convert the block mask into a dense boolean mask.

        Returns:
            A dense boolean mask of shape (B, H, Q_LEN, KV_LEN)
        """
        num_q_blocks = self.kv_num_blocks.shape[-1]

        mask = jnp.full(
            (self.B, self.H, self.Q_LEN, self.KV_LEN), False, dtype=jnp.bool
        )
        mask = rearrange(
            mask,
            "B H (Q Qb) (KV KVb) ->B H Q KV Qb KVb",
            Qb=self.Q_BLOCK_SIZE,
            KVb=self.KV_BLOCK_SIZE,
        )

        # add in the partial blocks
        def partial_loop_fn(b, h, q_block_idx, sparse_idx, mask):
            kv_block_idx = self.kv_indices[b, h, q_block_idx, sparse_idx]
            partial_block = self.get_mask_for_partial_block(
                b, h, q_block_idx, kv_block_idx
            )
            return mask.at[b, h, q_block_idx, kv_block_idx].set(partial_block)

        mask = nested_fori_loop(
            lowers=(0, 0, 0, 0),
            uppers=(
                self.B,
                self.H,
                num_q_blocks,
                lambda b, h, q_block_idx: self.kv_num_blocks[b, h, q_block_idx],
            ),
            body_fun=partial_loop_fn,
            init_val=mask,
        )

        # now add in the full blocks
        def full_loop_fn(b, h, q_block_idx, sparse_idx, mask):
            kv_block_idx = self.full_kv_indices[b, h, q_block_idx, sparse_idx]
            # full_block = jnp.full((self.Q_BLOCK_SIZE, self.KV_BLOCK_SIZE), True)
            return mask.at[b, h, q_block_idx, kv_block_idx].set(True)  # full_block)

        mask = nested_fori_loop(
            lowers=(0, 0, 0, 0),
            uppers=(
                self.B,
                self.H,
                num_q_blocks,
                lambda b, h, q_block_idx: self.full_kv_num_blocks[b, h, q_block_idx],
            ),
            body_fun=full_loop_fn,
            init_val=mask,
        )

        mask = rearrange(mask, "B H Q KV Qb KVb ->B H (Q Qb) (KV KVb)").astype(
            jnp.int32
        )

        return mask

    def get_full_blocks(self) -> Array:
        """
        Get the block representation of full (all-ones) blocks.

        Returns:
            A dense matrix where 1 indicates a full block and 0 indicates no block
        """
        ROWS = self.full_kv_num_blocks.shape[-1]
        COLS = self.full_q_num_blocks.shape[-1]
        return get_dense_from_kv_blocks(
            self.B, self.H, ROWS, COLS, self.full_kv_num_blocks, self.full_kv_indices
        )

    def get_partial_blocks(self) -> Array:
        """
        Get the block representation of partial blocks.

        Returns:
            A dense matrix where 1 indicates a partial block and 0 indicates no block
        """
        ROWS = self.kv_num_blocks.shape[-1]
        COLS = self.q_num_blocks.shape[-1]
        return get_dense_from_kv_blocks(
            self.B, self.H, ROWS, COLS, self.kv_num_blocks, self.kv_indices
        )

    def to_dense(self) -> Array:
        """
        Convert the block mask to a dense mask.
        """

        ROWS = self.kv_num_blocks.shape[-1]
        COLS = self.q_num_blocks.shape[-1]

        return self.get_full_blocks() + self.get_partial_blocks()

    @classmethod
    def full_mask(cls, B, H, Q_LEN, KV_LEN, BLOCK_SIZE):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = extract_block_size(BLOCK_SIZE)

        KV_BLOCKS = KV_LEN // KV_BLOCK_SIZE
        Q_BLOCKS = Q_LEN // Q_BLOCK_SIZE

        return cls.from_kv_blocks(
            kv_num_blocks=jnp.zeros((B, H, Q_BLOCKS), dtype=jnp.int32),
            kv_indices=jnp.ones((B, H, Q_BLOCKS, 1), dtype=jnp.int32),
            full_kv_num_blocks=jnp.full((B, H, Q_BLOCKS), KV_BLOCKS, dtype=jnp.int32),
            full_kv_indices=jnp.broadcast_to(
                jnp.arange(KV_BLOCKS), (B, H, Q_BLOCKS, KV_BLOCKS)
            ),
            BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
            mask_mod=lambda b, h, q_idx, kv_idx: True,
            seq_lengths=(Q_LEN, KV_LEN),
        )

    @classmethod
    def causal_mask(cls, B, H, Q_LEN, KV_LEN, BLOCK_SIZE):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = extract_block_size(BLOCK_SIZE)
        KV_BLOCKS = KV_LEN // KV_BLOCK_SIZE
        Q_BLOCKS = Q_LEN // Q_BLOCK_SIZE

        full_blocks = jnp.broadcast_to(
            jnp.tri(Q_BLOCKS, KV_BLOCKS, -1, dtype=jnp.int32),
            (B, H, Q_BLOCKS, KV_BLOCKS),
        )
        partial_blocks = jnp.broadcast_to(
            jnp.eye(Q_BLOCKS, KV_BLOCKS, dtype=jnp.int32), (B, H, Q_BLOCKS, KV_BLOCKS)
        )

        kv_num_blocks, kv_indices = get_sparse_kv_data_from_blocks(B, H, partial_blocks)
        q_num_blocks, q_indices = get_sparse_q_data_from_blocks(B, H, partial_blocks)

        full_kv_num_blocks, full_kv_indices = get_sparse_kv_data_from_blocks(
            B, H, full_blocks
        )

        full_q_num_blocks, full_q_indices = get_sparse_q_data_from_blocks(
            B, H, full_blocks
        )

        return BlockMask(
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            Q_BLOCK_SIZE=Q_BLOCK_SIZE,
            KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            mask_mod=lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
        )

    @classmethod
    def from_blocks(
        cls,
        partial_blocks,
        full_blocks,
        BLOCK_SIZE: int | Tuple[int, int],
        mask_mod: Callable[[Array, Array, Array, Array], Array],
    ):
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = extract_block_size(BLOCK_SIZE)

        B = partial_blocks.shape[0]
        H = partial_blocks.shape[1]

        # remove the full blocks if they were left in the partial blocks.
        partial_blocks = jnp.where(
            full_blocks, jnp.zeros_like(partial_blocks), partial_blocks
        )

        NUM_Q_BLOCKS = partial_blocks.shape[2]
        NUM_KV_BLOCKS = partial_blocks.shape[3]

        Q_LEN = NUM_Q_BLOCKS * Q_BLOCK_SIZE
        KV_LEN = NUM_KV_BLOCKS * KV_BLOCK_SIZE

        kv_num_blocks, kv_indices = get_sparse_kv_data_from_blocks(B, H, partial_blocks)
        q_num_blocks, q_indices = get_sparse_q_data_from_blocks(B, H, partial_blocks)

        full_kv_num_blocks, full_kv_indices = get_sparse_kv_data_from_blocks(
            B, H, full_blocks
        )
        full_q_num_blocks, full_q_indices = get_sparse_q_data_from_blocks(
            B, H, full_blocks
        )

        return cls(
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            Q_BLOCK_SIZE=Q_BLOCK_SIZE,
            KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            mask_mod=mask_mod,
        )

    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Array,
        kv_indices: Array,
        full_kv_num_blocks: Array,
        full_kv_indices: Array,
        BLOCK_SIZE: int,
        mask_mod: Callable[[Array, Array, Array, Array], Array],
        seq_lengths: Tuple[int, int] = None,
    ) -> "BlockMask":
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = extract_block_size(BLOCK_SIZE)

        B = kv_num_blocks.shape[0]
        H = kv_num_blocks.shape[1]

        if seq_lengths is None:
            Q_LEN = kv_indices.shape[2] * Q_BLOCK_SIZE

            def get_max_index(b, h, i, num_blocks, indices):
                indices = jnp.arange(num_blocks.shape[2], dtype=jnp.int32)
                return jnp.max(
                    jnp.where(
                        indices < num_blocks[b, h, i], indices[b, h, i, :], -jnp.inf
                    ).astype(jnp.int32)
                )

            max_kv_index = jnp.max(
                array_from_coords(
                    shape=(B, H, kv_indices.shape[2]),
                    fn=lambda b, h, i: get_max_index(
                        b, h, i, kv_num_blocks, kv_indices
                    ),
                )
            )

            max_full_kv_index = jnp.max(
                array_from_coords(
                    shape=(B, H, full_kv_indices.shape[2]),
                    fn=lambda b, h, i: get_max_index(
                        b, h, i, full_kv_num_blocks, full_kv_indices
                    ),
                )
            )
            KV_LEN = jnp.maximum(max_kv_index, max_full_kv_index) * KV_BLOCK_SIZE
        else:
            Q_LEN, KV_LEN = seq_lengths

        NUM_Q_BLOCKS = Q_LEN // Q_BLOCK_SIZE
        NUM_KV_BLOCKS = KV_LEN // KV_BLOCK_SIZE

        partial_mask = get_dense_from_kv_blocks(
            B, H, NUM_Q_BLOCKS, NUM_KV_BLOCKS, kv_num_blocks, kv_indices
        )

        full_mask = get_dense_from_kv_blocks(
            B, H, NUM_Q_BLOCKS, NUM_KV_BLOCKS, full_kv_num_blocks, full_kv_indices
        )

        return cls.from_blocks(
            partial_blocks=partial_mask,
            full_blocks=full_mask,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=mask_mod,
        )


def get_partial_block(
    mask_mod, b, h, block_q_idx, block_kv_idx, Q_BLOCK_SIZE, KV_BLOCK_SIZE, B, H
):
    """
    Get a partial block mask for the specified indices.

    Args:
        mask_mod: Function that computes the mask for a single position
        b: batch index
        h: head index
        block_q_idx: query block index
        block_kv_idx: key-value block index
        Q_BLOCK_SIZE: Size of query blocks
        KV_BLOCK_SIZE: Size of key-value blocks
        B: Batch size
        H: Number of heads

    Returns:
        A boolean mask of shape (Q_BLOCK_SIZE, KV_BLOCK_SIZE) for the specified block
    """
    block_q_start = block_q_idx * Q_BLOCK_SIZE
    block_kv_start = block_kv_idx * KV_BLOCK_SIZE

    block_mask_mod = lambda q_idx, kv_idx: mask_mod(
        b, h, block_q_start + q_idx, block_kv_start + kv_idx
    )

    return array_from_coords(shape=(Q_BLOCK_SIZE, KV_BLOCK_SIZE), fn=block_mask_mod)


def get_sparse_kv_data_from_blocks(B, H, blocks: Array):
    """
    Convert a dense block mask into sparse matrix block data along the key-value (row) axis.

    Args:
        B: Batch size
        H: Number of heads
        blocks: Dense block mask

    Returns:
        Tuple of (kv_num_blocks, kv_indices) where:
        - kv_num_blocks: Number of blocks for each key-value position
        - kv_indices: Indices of the blocks for each key-value position
    """
    kv_num_blocks = einsum(blocks, "b h kv q -> b h kv")  # [b h kv q] -> [b h kv]
    kv_indices = jax.vmap(lambda x: jnp.argsort(-x))(blocks)

    return kv_num_blocks, kv_indices


def get_sparse_q_data_from_blocks(B, H, blocks: Array):
    """
    Convert a dense block mask into sparse query block data.

    Args:
        B: Batch size
        H: Number of heads
        num_blocks_in_col: Number of blocks in each column
        num_blocks_in_row: Number of blocks in each row
        blocks: Dense block mask

    Returns:
        Tuple of (q_num_blocks, q_indices) where:
        - q_num_blocks: Number of blocks for each query position
        - q_indices: Indices of the blocks for each query position
    """
    return get_sparse_kv_data_from_blocks(
        B, H, rearrange(blocks, "b h q kv -> b h kv q")
    )


def create_block_mask(
    mask_mod: Callable[[Array, Array, Array, Array], Array],
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: int,
):
    """
    Create a block mask for the given mask_mod.
    """

    Q_BLOCK_SIZE, KV_BLOCK_SIZE = extract_block_size(BLOCK_SIZE)

    num_blocks_in_col = Q_LEN // Q_BLOCK_SIZE
    num_blocks_in_row = KV_LEN // KV_BLOCK_SIZE

    def is_partial_block(b, h, block_q_idx, block_kv_idx):
        block = get_partial_block(
            mask_mod, b, h, block_q_idx, block_kv_idx, Q_BLOCK_SIZE, KV_BLOCK_SIZE, B, H
        )
        return (jnp.any(block) ^ jnp.all(block)) * 1

    def is_full_block(b, h, block_q_idx, block_kv_idx):
        block = get_partial_block(
            mask_mod, b, h, block_q_idx, block_kv_idx, Q_BLOCK_SIZE, KV_BLOCK_SIZE, B, H
        )
        return (jnp.all(block)) * 1

    # create partial block mask:
    partial_block_mask = array_from_coords(
        shape=(B, H, num_blocks_in_col, num_blocks_in_row), fn=is_partial_block
    )

    # create full block mask:
    full_block_mask = array_from_coords(
        shape=(B, H, num_blocks_in_col, num_blocks_in_row), fn=is_full_block
    )

    return BlockMask.from_blocks(
        partial_blocks=partial_block_mask,
        full_blocks=full_block_mask,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
    )


def get_dense_from_kv_blocks(
    B, H, NUM_Q_BLOCKS, NUM_KV_BLOCKS, kv_num_blocks, kv_indices
) -> Array:
    """
    Convert sparse key-value block data into a dense block mask.

    Args:
        B: Batch size
        H: Number of heads
        NUM_Q_BLOCKS: Number of query blocks
        NUM_KV_BLOCKS: Number of key-value blocks
        kv_num_blocks: Number of blocks for each key-value position
        kv_indices: Indices of the blocks for each key-value position

    Returns:
        A dense matrix where 1 indicates a block and 0 indicates no block
    """

    ROWS = NUM_Q_BLOCKS
    COLS = NUM_KV_BLOCKS

    data = jnp.zeros((B, H, ROWS, COLS), dtype=jnp.int32)

    def set_value(b, h, i, j, acc):
        return acc.at[b, h, i, kv_indices[b, h, i, j]].set(1)

    def get_upper(b, h, i):
        result = kv_num_blocks[b, h, i]
        return result

    return nested_fori_loop(
        lowers=(0, 0, 0, 0),
        uppers=(B, H, ROWS, get_upper),
        body_fun=set_value,
        init_val=data,
    )
