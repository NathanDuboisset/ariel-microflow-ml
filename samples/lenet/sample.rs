use microflow::buffer::Buffer4D;
use nalgebra::SMatrix;

const BG: [i8; 1] = [-128_i8];
const FG: [i8; 1] = [127_i8];

pub fn digit_0() -> Buffer4D<i8, 1, 28, 28, 1> {
    type Img = SMatrix<[i8; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        if (r == 4 || r == 23) && (10..=17).contains(&c) {
            return FG;
        }
        if (r == 5 || r == 22) && (8..=19).contains(&c) {
            return FG;
        }
        if (r == 6 || r == 7 || r == 8 || r == 19 || r == 20 || r == 21) && (7..=20).contains(&c) {
            return FG;
        }
        if (9..=18).contains(&r) && ((6..=8).contains(&c) || (19..=21).contains(&c)) {
            return FG;
        }
        BG
    });
    [img]
}

pub fn digit_1() -> Buffer4D<i8, 1, 28, 28, 1> {
    type Img = SMatrix<[i8; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        if (5..=22).contains(&r) && (13..=15).contains(&c) {
            return FG;
        }
        if (r == 6 || r == 7) && (12..=16).contains(&c) {
            return FG;
        }
        if r == 23 && (10..=18).contains(&c) {
            return FG;
        }
        BG
    });
    [img]
}
