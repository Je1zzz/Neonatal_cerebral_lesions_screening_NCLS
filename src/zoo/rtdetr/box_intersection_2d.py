'''
torch implementation of 2d oriented box intersection

author: lanxiao li
2020.8
'''
import torch
from cuda_op.cuda_ext import sort_v
EPSILON = 1e-8

def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor):
    """
    Find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point.

    Args:
        corners1 (torch.Tensor): B, N, 4, 2
        corners2 (torch.Tensor): B, M, 4, 2

    Returns:
        intersections (torch.Tensor): B, N, M, 4, 4, 2
        mask (torch.Tensor): B, N, M, 4, 4; bool
    """
    # Build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)  # B, N, 4, 4
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)  # B, M, 4, 4

    # Pair each edge from the boxes
    line1_ext = line1.unsqueeze(2).unsqueeze(4)  # B, N, 1, 4, 1, 4
    line2_ext = line2.unsqueeze(1).unsqueeze(3)  # B, 1, M, 1, 4, 4

    line1_ext = line1_ext.expand(-1, -1, corners2.size(1), -1, 4, -1)  # B, N, M, 4, 4, 4
    line2_ext = line2_ext.expand(-1, corners1.size(1), -1, 4, -1, -1)  # B, N, M, 4, 4, 4

    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]

    # Math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / (num + EPSILON)
    t[num == 0] = -1.0  # Handle collinear lines by setting t out of bounds
    mask_t = (t > 0) & (t < 1)  # Intersection on line segment 1

    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / (num + EPSILON)
    u[num == 0] = -1.0  # Handle collinear lines by setting u out of bounds
    mask_u = (u > 0) & (u < 1)  # Intersection on line segment 2

    mask = mask_t & mask_u  # Intersection on both segments

    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)  # B, N, M, 4, 4, 2
    intersections = intersections * mask.float().unsqueeze(-1)

    return intersections, mask  # [B, N, M, 4, 4, 2], [B, N, M, 4, 4]

def box1_in_box2(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, M, 4, 2)

    Returns:
        c1_in_2: (B, N,M, 4) Bool
    """
    a = corners2[:, :, 0:1, :].unsqueeze(1)  # (B, 1, M, 1, 2)
    b = corners2[:, :, 1:2, :].unsqueeze(1)  # (B, 1, M, 1, 2)
    d = corners2[:, :, 3:4, :].unsqueeze(1)  # (B, 1, M, 1, 2)

    ab = b - a                              # (B, 1, M, 1, 2)
    am = corners1.unsqueeze(2) - a          # (B, N, M, 4, 2)
    ad = d - a                              # (B, 1, M, 1, 2)

    p_ab = torch.sum(ab * am, dim=-1)  # (B, N, M, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, 1, M, 1)
    p_ad = torch.sum(ad * am, dim=-1)  # (B, N, M, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, 1, M, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > -1e-6) * (p_ab / norm_ab < 1 + 1e-6)  # (B, N, M, 4)
    cond2 = (p_ad / norm_ad > -1e-6) * (p_ad / norm_ad < 1 + 1e-6)  # (B, N, M, 4)
    return cond1*cond2

def box_in_box_th(corners1:torch.Tensor, corners2:torch.Tensor):
    """check if corners of two boxes lie in each other

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, M, 4, 2)

    Returns:
        c1_in_2: (B, N,M, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (B, N,M, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1).transpose(1, 2)
    return c1_in_2, c2_in_1

def build_vertices(corners1:torch.Tensor, corners2:torch.Tensor, 
                c1_in_2:torch.Tensor, c2_in_1:torch.Tensor, 
                inters:torch.Tensor, mask_inter:torch.Tensor):
    """find vertices of intersection area

    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, M, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (B, N, M, 4)
        c2_in_1 (torch.Tensor): Bool, (B, N, M, 4)
        inters (torch.Tensor): (B, N, M, 4, 4, 2)
        mask_inter (torch.Tensor): (B, N, M, 4, 4)
    
    Returns:
        vertices (torch.Tensor): (B, N, M, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (B, N, M, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0). 
    # can be used as trick
    B, N, M, _, _, _ = inters.size()
    vertices = torch.cat([corners1.unsqueeze(2).expand(-1, -1, M, -1, -1),
                          corners2.unsqueeze(1).expand(-1, N, -1, -1, -1),
                          inters.view(B, N, M, -1, 2)], dim=3)  # (B, N, M, 24, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view(B, N, M, -1)], dim=3)  # Bool (B, N, M, 24)
    return vertices, mask

def sort_indices(vertices:torch.Tensor, mask:torch.Tensor):
    """[summary]

    Sort the indices to form the polygon
    Args:
        vertices (torch.Tensor): float (B, N, M, 24, 2)
        mask (torch.Tensor): bool (B, N, M, 24)
    Returns:
        sorted_index: bool (B, N, M, 9)
    
    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    num_valid = torch.sum(mask.int(), dim=3).int()  # (B, N, M)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=3, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # Normalization for easier sorting
    return sort_v(vertices_normalized, mask, num_valid).long()

def calculate_area(idx_sorted:torch.Tensor, vertices:torch.Tensor):
    """calculate area of intersection

    Calculate the area of the intersection polygon
    Args:
        idx_sorted (torch.Tensor): (B, N, M, 9)
        vertices (torch.Tensor): (B, N, M, 24, 2)
    Returns:
        area: (B, N, M), area of intersection
        selected: (B, N, M, 9, 2), vertices of polygon with zero padding
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat(1, 1, 1, 1, 2)
    selected = torch.gather(vertices, 3, idx_ext)
    total = selected[..., 0:-1, 0] * selected[..., 1:, 1] - selected[..., 0:-1, 1] * selected[..., 1:, 0]
    total = torch.sum(total, dim=3)
    area = torch.abs(total) / 2
    return area, selected

def oriented_box_intersection_2d(corners1:torch.Tensor, corners2:torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Calculate intersection area of 2D rectangles
    Args:
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners2 (torch.Tensor): (B, M, 4, 2)
    Returns:
        area: (B, N, M), area of intersection
        selected: (B, N, M, 9, 2), vertices of polygon with zero padding 
    """
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)
