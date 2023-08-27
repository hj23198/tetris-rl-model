use pyo3;
use pyo3::prelude::*;


#[pymodule]
fn rstris(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Env>()?;
    Ok(())
}

#[pyclass]
struct Env {
    game: Game,
}
#[pymethods]
impl Env {
    #[new]
    fn new() -> Self {
        return Env { game: Game::new() };
    }
    fn display(&self) {
        self.game.board.display_with_attatched()
    }

    ///Weight order: DASL(0), L(1), DASR(2), R(3), HD(4), SD(5), RCCW(6), RCW(7), H(8)
    fn step(&mut self, selected: f32) -> ((Vec<Vec<bool>>, [i16; 11], [bool; 8], [[bool; 7]; 5]), u32, bool) {
        match selected as u8 {
            0 => self.game.das_left(),
            1 => self.game.left(),
            2 => self.game.das_right(),
            3 => self.game.right(),
            4 => self.game.hard_drop(),
            5 => self.game.soft_drop(),
            6 => self.game.rotate_counterclockwise(),
            7 => self.game.rotate_clockwise(),
            8 => self.game.hold(),
            _ => panic!("Recived invalid move"),
        }
        self.game.board.clear_lines();
        let mut reward = self.game.board.lines_sent as u32 * 3 + (self.game.board.pieces_placed * 4 / 10);
        if self.game.is_terminated {
            reward = 0
        }

        return (
            self.get_state(),
            reward,
            self.game.is_terminated,
        );
    }

    // game.board.state
    // piece type [s, z, j, l, t, i, o]
    // hold state [empty, z, h, l, t, i, i]
    fn get_state(&self) -> (Vec<Vec<bool>>, [i16; 11], [bool; 8], [[bool; 7]; 5]) {
        let mut current_piece: [i16; 11] = [0; 11];
        let idx = match self.game.board.piece.name.as_str() {
            "S" => 0,
            "Z" => 1,
            "J" => 2,
            "L" => 3,
            "T" => 4,
            "I" => 5,
            "O" => 6,
            _ => panic!("Ran into nonexistant piece"),
        };
        current_piece[idx] = 1;
        current_piece[7] = self.game.board.piece.location.x as i16;
        current_piece[8] = self.game.board.piece.location.y as i16;
        current_piece[9] = self.game.board.lines_sent as i16;
        current_piece[10] = self.game.board.pieces_placed as i16;



        let mut hold_piece: [bool; 8] = [false; 8];
        let idx = match &self.game.board.hold_piece {
            Some(x) => match x.name.as_str() {
                "S" => 1,
                "Z" => 2,
                "J" => 3,
                "L" => 4,
                "T" => 5,
                "I" => 6,
                "O" => 7,
                _ => panic!("Ran into nonexistant piece"),
            },
            None => 0
        };
        hold_piece[idx] = true;

        let mut queue = [[false; 7]; 5];
        for i in 0..5 {
            let idx = match self.game.board.bag.queue[i].as_str() {
                "S" => 0,
                "Z" => 1,
                "J" => 2,
                "L" => 3,
                "T" => 4,
                "I" => 5,
                "O" => 6,
                _ => panic!("Ran into nonexistant piece"),
            };
            queue[i][idx] = false;
        }

        let board_state: Vec<Vec<bool>> = self.game.board.state.clone();
        return (board_state, current_piece, hold_piece, queue);
    }

    fn get_attached_state(&self) -> Vec<Vec<bool>> {
        let mut display = self.game.board.state.clone();
        let ysize = self.game.board.piece.state.len();
        let xsize = self.game.board.piece.state[0].len();
        for y in 0..ysize {
            for x in 0..xsize {
                if self.game.board.piece.state[y][x] {
                    display[(self.game.board.piece.location.y + y as i8) as usize]
                        [(self.game.board.piece.location.x + x as i8) as usize] = true;
                }
            }
        }
        return display
    }
}

struct Bag {
    queue: Vec<String>,
    starting_pieces: Vec<String>,
    starting_location: Point,
    current_pieces: Vec<String>,
}

impl Bag {
    fn new(starting_pieces: Vec<String>, location: Point, queue_length: u8) -> Bag {
        let mut temp_bag = starting_pieces.clone();
        let mut queue = Vec::new();
        for _ in 0..queue_length {
            if temp_bag.len() == 0 {
                temp_bag = starting_pieces.clone();
            }
            let index = (rand::random::<f32>() * temp_bag.len() as f32).floor() as usize;
            let piece = temp_bag[index].clone();
            temp_bag.remove(index);
            queue.push(piece);
        }

        return Bag {
            starting_pieces: starting_pieces,
            current_pieces: temp_bag,
            starting_location: location,
            queue: queue,
        };
    }

    fn grab(&mut self) -> Piece {
        if self.current_pieces.len() == 0 {
            self.current_pieces = self.starting_pieces.clone();
        }

        let index = (rand::random::<f32>() * self.current_pieces.len() as f32).floor() as usize;
        let piece = self.current_pieces[index].clone();
        self.current_pieces.remove(index);
        self.queue.push(piece);

        let selected_piece = self.queue[0].clone();
        self.queue.remove(0);

        return Piece::new(selected_piece, self.starting_location.clone());
    }
}

pub struct Game {
    board: Board,
    is_terminated: bool,
}

impl Game {
    pub fn new() -> Game {
        // The board height does not matter, the starting piece location does instead
        let board = Board::new(10, 30);
        return Game {
            board: board,
            is_terminated: false,
        };
    }

    pub fn das_left(&mut self) {
        while self.board.can_fit_offset(-1, 0) {
            self.board.piece.location.x -= 1;
        }
    }

    pub fn left(&mut self) {
        if self.board.can_fit_offset(-1, 0) {
            self.board.piece.location.x -= 1;
        }
    }

    pub fn das_right(&mut self) {
        while self.board.can_fit_offset(1, 0) {
            self.board.piece.location.x += 1;
        }
    }

    pub fn right(&mut self) {
        if self.board.can_fit_offset(1, 0) {
            self.board.piece.location.x += 1;
        }
    }

    pub fn hard_drop(&mut self) {
        while self.board.can_fit_offset(0, -1) {
            self.board.piece.location.y -= 1;
        }
        self.board.attatch_piece();
        self.board.piece = self.board.bag.grab();
        self.is_terminated = self.board.state[21][4] || self.board.state[22][4] || self.board.state[21][5] || self.board.state[22][5];
        self.board.pieces_placed += 1
    }

    pub fn soft_drop(&mut self) {
        while self.board.can_fit_offset(0, -1) {
            self.board.piece.location.y -= 1;
        }
    }

    pub fn rotate_counterclockwise(&mut self) {
        self.board.rotate_piece(false)
    }

    pub fn rotate_clockwise(&mut self) {
        self.board.rotate_piece(true)
    }

    /// Swaps current piece and held piece
    pub fn hold(&mut self) {
        match self.board.hold_piece {
            Some(_) => {
                let temp = Piece::new(self.board.piece.name.clone(), self.board.bag.starting_location.clone());
                self.board.piece = self.board.hold_piece.clone().unwrap();
                self.board.hold_piece = Some(temp);
            }
            None => {
                let mut piece = Piece::new(self.board.piece.name.clone(), self.board.bag.starting_location.clone());
                piece.location = self.board.bag.starting_location.clone();
                self.board.hold_piece = Some(piece);
                self.board.piece = self.board.bag.grab();
            }
        }
    }
}

struct Board {
    width: usize,
    height: usize,
    state: Vec<Vec<bool>>,
    piece: Piece,
    hold_piece: Option<Piece>,
    bag: Bag,
    b2b_chain: u8,
    combo: u8,
    lines_sent: u16,
    pieces_placed: u32
}

impl Board {
    fn new(width: usize, height: usize) -> Board {
        let state: Vec<Vec<bool>> = vec![vec![false; width]; height];
        let mut bag = Bag::new(
            vec![
                "I".to_string(),
                "J".to_string(),
                "L".to_string(),
                "O".to_string(),
                "S".to_string(),
                "T".to_string(),
                "Z".to_string(),
            ],
            Point::new(3, 20),
            5,
        );
        let first_piece = bag.grab();
        return Board {
            width: width,
            height: height,
            bag: bag,
            piece: first_piece,
            hold_piece: None,
            b2b_chain: 0,
            combo: 0,
            lines_sent: 0,
            state: state,
            pieces_placed: 0
        };
    }

    /// Clears full lines from board. Returns the amount of lines cleared.
    fn clear_lines(&mut self) -> u8 {
        let mut lines_removed: u8 = 0;
        for i in (0..self.state.len()).rev() {
            if self.state[i].iter().all(|&x| x) {
                self.state.remove(i);
                lines_removed += 1;
            }
        }
        self.lines_sent += lines_removed as u16;

        for _ in 0..lines_removed {
            self.state.push(vec![false; self.width])
        }
        return lines_removed;
    }

    /// Attempts to rotate current piece
    fn rotate_piece(&mut self, is_cw: bool) {
        if self.piece.name == "O" {
            return;
        }

        let output: (Vec<Vec<bool>>, [[i8; 2]; 5]);
        if is_cw {
            output = self.piece.try_rotate_cw();
        } else {
            output = self.piece.try_rotate_ccw();
        }

        let state: Vec<Vec<bool>> = output.0;
        let kick_table: [[i8; 2]; 5] = output.1;
        let mut chosen_kick: Option<[i8; 2]> = None;

        for i in 0..5 {
            if self.can_fit(&state, kick_table[i][0], kick_table[i][1]) {
                chosen_kick = Some(kick_table[i]);
                break;
            }
        }

        match chosen_kick {
            None => return,
            Some(kick) => {
                self.piece.location.x = kick[0];
                self.piece.location.y = kick[1];

                if is_cw {
                    self.piece.do_rotate_cw();
                } else {
                    self.piece.do_rotate_ccw();
                }
            }
        }
    }

    /// Places current piece on board state. Does NOT generate a new current piece. Does NOT check if the piece can fit.
    fn attatch_piece(&mut self) {
        if self.can_fit_offset(0, 0) == false {
            println!("WARNING: attached piece does not fit");
        }
        let ysize = self.piece.state.len();
        let xsize = self.piece.state[0].len();

        for y in 0..ysize {
            for x in 0..xsize {
                if self.piece.state[y][x] {
                    self.state[(self.piece.location.y + y as i8) as usize]
                        [(self.piece.location.x + x as i8) as usize] = true;
                }
            }
        }
    }

    ///ASCII art for board
    fn display(&self) {
        for i in 0..self.height {
            for j in 0..self.width {
                if self.state[self.height - i - 1][j] {
                    print!("[]")
                } else {
                    print!("  ")
                }
                // if self.lines[self.height - i - 1][j] {
                //     print!("1")
                // } else {
                //     print!("0")
                // }
            }
            println!()
        }
    }

    fn display_with_attatched(&self) {
        let mut display = self.state.clone();
        let ysize = self.piece.state.len();
        let xsize = self.piece.state[0].len();

        for y in 0..ysize {
            for x in 0..xsize {
                if self.piece.state[y][x] {
                    display[(self.piece.location.y + y as i8) as usize]
                        [(self.piece.location.x + x as i8) as usize] = true;
                }
            }
        }

        for i in 0..self.height {
            for j in 0..self.width {
                if display[self.height - i - 1][j] {
                    print!("[]")
                } else {
                    print!("  ")
                }
                // if display[self.height - i - 1][j] {
                //     print!("1")
                // } else {
                //     print!("0")
                // }
            }
            println!()
        }
    }

    /// Returns true if piece can be placed at location
    fn can_fit(&self, state: &Vec<Vec<bool>>, x: i8, y: i8) -> bool {
        let ysize = state.len();
        let xsize = state[0].len();

        for ymod in 0..ysize {
            for xmod in 0..xsize {
                if state[ymod][xmod] == true {
                    let new_x = x as i8 + xmod as i8;
                    let new_y = y as i8 + ymod as i8;

                    if new_x < 0 || self.width as i8 - 1 < new_x {
                        return false;
                    }

                    if new_y < 0 || self.height as i8 - 1 < new_y {
                        return false;
                    }

                    if self.state[new_y as usize][new_x as usize] == true {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    /// Returns true if current piece can be placed its current location with an offset
    fn can_fit_offset(&self, dx: i8, dy: i8) -> bool {
        self.can_fit(
            &self.piece.state,
            self.piece.location.x as i8 + dx,
            self.piece.location.y as i8 + dy,
        )
    }
}

#[derive(Clone)]
struct Piece {
    name: String,
    state: Vec<Vec<bool>>,
    rotation: u8,
    location: Point,
}

//TODO: find location to spawn new piece
impl Piece {
    fn new(piece_type: String, location: Point) -> Piece {
        Piece {
            name: piece_type.clone(),
            state: generate_piece(&piece_type),
            rotation: 0,
            location: location,
        }
    }

    /// Returns vector representation of clockwise rotated piece and list of new coordinates
    fn try_rotate_cw(&self) -> (Vec<Vec<bool>>, [[i8; 2]; 5]) {
        let rotation_state = rotate_piece(&self.state, true);
        //let rotation_amount = (self.rotation + 1) % 4;  //TODO REMOVE?
        let mut kick_table = get_rotation_offset(self.name.to_string() == "I", true, self.rotation);
        for i in 0..5 {
            kick_table[i][0] += self.location.x as i8;
            kick_table[i][1] += self.location.y as i8;
        }
        return (rotation_state, kick_table);
    }

    /// Rotates piece state after clockwise rotation
    fn do_rotate_cw(&mut self) {
        self.state = rotate_piece(&self.state, true);
        self.rotation = (self.rotation + 1) % 4
    }

    /// Returns vector representation of counter clockwise rotated piece and list of new coordinates
    fn try_rotate_ccw(&self) -> (Vec<Vec<bool>>, [[i8; 2]; 5]) {
        let rotation_state = rotate_piece(&self.state, false);
        //let rotation_amount = (self.rotation + 3) % 4; //TODO REMOVE?
        let mut kick_table =
            get_rotation_offset(self.name.to_string() == "I", false, self.rotation);
        for i in 0..5 {
            kick_table[i][0] += self.location.x as i8;
            kick_table[i][1] += self.location.y as i8;
        }
        return (rotation_state, kick_table);
    }

    /// Rotates piece state after counter clockwise rotation
    fn do_rotate_ccw(&mut self) {
        self.state = rotate_piece(&self.state, false);
        self.rotation = (self.rotation + 3) % 4
    }
}

/// Returns coodinate shift for a rotation based on SRS rotation system
/// from https://tetris.wiki/Super_Rotation_System
fn get_rotation_offset(is_i: bool, is_cw: bool, piece_orientation: u8) -> [[i8; 2]; 5] {
    if is_i {
        if is_cw {
            match piece_orientation {
                0 => return [[0, 0], [-2, 0], [1, 0], [-2, 1], [1, 2]], // 0 -> r
                1 => return [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]], // r -> 2
                2 => return [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]], // 2 -> l
                3 => return [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]], // l -> 0
                _ => panic!("Given invalid orientation value '{}'", piece_orientation),
            }
        } else {
            match piece_orientation {
                0 => return [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]], // 0 -> l
                1 => return [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]], // r -> 0
                2 => return [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]], // 2 -> r
                3 => return [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]], // l -> 2
                _ => panic!("Given invalid orientation value '{}'", piece_orientation),
            }
        }
    } else if is_cw {
        match piece_orientation {
            0 => return [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]], // 0 -> r
            1 => return [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],     // r -> 2
            2 => return [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],    // 2 -> l
            3 => return [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],  // l -> 0
            _ => panic!("Given invalid orientation value '{}'", piece_orientation),
        }
    } else {
        match piece_orientation {
            0 => return [[0, 0], [1, 0], [1, 1], [0, -1], [1, -2]], // 0 -> l
            1 => return [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],  // r -> 0
            2 => return [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]], // 2 -> r
            3 => return [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]], // l -> 2
            _ => panic!("Given invalid orientation value '{}'", piece_orientation),
        }
    }
}

/// Only works for nxn matricies.
/// Rotates given piece state into new piece state. Point is defined as the coordinates of the
/// bottom left the piece's bounding box, where the center of rotation is considered to be (0, 0) on
/// the coordinate plane.
fn rotate_piece(current_state: &Vec<Vec<bool>>, is_cw: bool) -> Vec<Vec<bool>> {
    let mut matrix: Vec<Vec<bool>> = current_state.clone();
    let length = matrix.len();
    if is_cw {
        for offset in 0..(length / 2) {
            for i in offset..(length - offset - 1) {
                let storage = matrix[offset][i];
                matrix[offset][i] = matrix[i][length - 1 - offset];
                matrix[i][length - 1 - offset] = matrix[length - 1 - offset][length - 1 - i];
                matrix[length - 1 - offset][length - 1 - i] = matrix[length - 1 - i][offset];
                matrix[length - 1 - i][offset] = storage;
            }
        }
    } else {
        for offset in 0..(length / 2) {
            for i in offset..(length - offset - 1) {
                let storage = matrix[offset][i];
                matrix[offset][i] = matrix[length - 1 - i][offset];
                matrix[length - 1 - i][offset] = matrix[length - 1 - offset][length - 1 - i];
                matrix[length - 1 - offset][length - 1 - i] = matrix[i][length - 1 - offset];
                matrix[i][length - 1 - offset] = storage;
            }
        }
    }
    return matrix;
}

/// Returns vector representation of piece at rotation state 0 for a given piece name
fn generate_piece(name: &String) -> Vec<Vec<bool>> {
    // Since 0 is considered to be the bottom left corner, the generated vecs appear to be flipped across the y axis
    match name.as_str() {
        "I" => {
            return vec![
                vec![false, false, false, false],
                vec![false, false, false, false],
                vec![true, true, true, true],
                vec![false, false, false, false],
            ]
        }
        "J" => {
            return vec![
                vec![false, false, false],
                vec![true, true, true],
                vec![true, false, false],
            ]
        }
        "L" => {
            return vec![
                vec![false, false, false],
                vec![true, true, true],
                vec![false, false, true],
            ]
        }
        "O" => {
            return vec![
                vec![false, false, false, false],
                vec![false, true, true, false],
                vec![false, true, true, false],
            ]
        }
        "S" => {
            return vec![
                vec![false, false, false],
                vec![true, true, false],
                vec![false, true, true],
            ]
        }
        "T" => {
            return vec![
                vec![false, false, false],
                vec![true, true, true],
                vec![false, true, false],
            ]
        }
        "Z" => {
            return vec![
                vec![false, false, false],
                vec![false, true, true],
                vec![true, true, false],
            ]
        }
        _ => panic!("Attempted to generate non existant piece"),
    }
}

fn get_lines_sent(
    b2b_combo: u16,
    combo: u8,
    lines_cleared: u16,
    is_tspin: bool,
    is_tspinmini: bool,
    is_pc: bool,
) {
    let mut lines_sent: u16;

    let b2b_level = match b2b_combo {
        0..=1 => 0,
        2..=3 => 1,
        4..=8 => 2,
        9..=24 => 3,
        _ => 4,
    };
}

#[derive(Clone)]
struct Point {
    x: i8,
    y: i8,
}

impl Point {
    fn new(x: i8, y: i8) -> Point {
        return Point { x: x, y: y };
    }
}
