#[macro_use]
extern crate gfx;
extern crate glutin;
extern crate gfx_window_glutin;
extern crate collada;
extern crate cgmath;
extern crate rusqlite;
extern crate png;

use std::path::Path;
use collada::document::ColladaDocument;
use rusqlite::Connection;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;

use cgmath:: {
    EuclideanSpace,
    Point3,
    Vector3,
    Matrix4,
    One,
    Zero,
    Rotation,
    // Quaternion,
};

pub fn main() {
    use gfx::traits::FactoryExt;
    use gfx::Device;
    use gfx::Factory;

    let mut conn = Connection::open(&Path::new("test.sqlite")).expect("failed to open sqlite file");
     // let mut conn = Connection::open_in_memory().unwrap();
     // create_table(&conn);
    // register_collada(&mut conn, 2, "assets/tree.dae", 2, "assets/tree.png",).expect("failed to register");

    let width = 1024;
    let height = 768;
    let builder = glutin::WindowBuilder::new()
        .with_title("PARTI")
        .with_dimensions(width, height)
        .with_vsync();
    let (window, mut device, mut factory, main_color, main_depth) =
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder);

    let pso = factory.create_pipeline_simple(
        include_bytes!("shader/150.glslv"),
        include_bytes!("shader/150.glslf"),
        pipe::new()
    ).expect("failed to create pipeline");

    let sampler_info = gfx::texture::SamplerInfo::new(
        gfx::texture::FilterMethod::Trilinear,
        gfx::texture::WrapMode::Clamp);
    let sampler = factory.create_sampler(sampler_info);

    let result = query_mesh(&conn, 1);
    let img = query_texture(&conn, 1);
    let tex_kind = gfx::texture::Kind::D2(img.width, img.height,
                                      gfx::texture::AaMode::Single);
    let (_, view) = factory.create_texture_immutable_u8::<gfx::format::Srgba8>(
                            tex_kind, &[&img.data]).expect("create texture failure");

    let mut entries = Vec::new();
    for vertex_data in result
    {
        let index_data: Vec<u32> = vertex_data.iter().enumerate().map(|(i, _)| i as u32).collect();
        let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, &index_data[..]);

        entries.push(Entry {
            slice: slice,
            vertex_buffer: vbuf,
            position: Vector3::zero(),
            front: Vector3::new(0.0, -1.0, 0.0)
        });
    }
    let mut camera = Camera::new(Point3::new(0.0, -7.0, 0.0),
                                 Point3::origin(),
                                 cgmath::PerspectiveFov  {
                                     fovy: cgmath::Rad(105.0f32.to_radians()),
                                     aspect: (width as f32) / (height as f32),
                                     near: 0.001,
                                     far: 1000.0,
                                 });

    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();
    // let r0 = Rotation::between_vectors(Vector3::new(0.0, -1.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

    'main: loop {
        for event in window.poll_events() {
            match event_handler(event) {
                GameCommand::Exit => break 'main,
                GameCommand::CameraMove(v) => { camera.translate(v); camera.update() },
                GameCommand::AvatorMoveW => { entries[0].position += Vector3::new(0.0, 1.0, 0.0); },
                GameCommand::AvatorMoveS => { entries[0].position -= Vector3::new(1.0, 0.0, 0.0); },
                GameCommand::AvatorMoveA => { entries[0].position += Vector3::new(1.0, 0.0, 0.0); },
                GameCommand::AvatorMoveD => { entries[0].position -= Vector3::new(1.0, 1.0, 0.0); },
                _ => {}
            }
        }

        encoder.clear(&main_color.clone(), CLEAR_COLOR);
        encoder.clear_depth(&main_depth, 1.0);

        for entry in entries.iter_mut() {
            let data = pipe::Data {
                vbuf: entry.vertex_buffer.clone(),
                u_model_view_proj: camera.projection.into(),
                u_model_view: camera.view.into(),
                u_light: [1.0, 0.0, -0.5f32],
                u_ambient_color: [0.01, 0.01, 0.01, 1.0],
                u_eye_direction: camera.direction().into(),
                u_texture: (view.clone(), sampler.clone()),
                u_translate: Matrix4::from_translation(entry.position).into(),
                out: main_color.clone(),
                out_depth: main_depth.clone()
            };
            encoder.draw(&entry.slice, &pso, &data);
        }
        encoder.flush(&mut device);
        window.swap_buffers();
        device.cleanup();
    }
}

enum GameCommand {
    CameraMove (Vector3<f32>),
    AvatorMoveW,
    AvatorMoveS,
    AvatorMoveA,
    AvatorMoveD,
    Exit,
    NOP
}

fn event_handler(ev :glutin::Event) -> GameCommand {
    match ev {
        glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
        glutin::Event::Closed => GameCommand::Exit,
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::L)) 
            => GameCommand::CameraMove(Vector3::new(0.5,0.0,0.0)),
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::H)) 
            => GameCommand::CameraMove(Vector3::new(-0.5,0.0,0.0)),
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::J)) 
            => GameCommand::CameraMove(Vector3::new(0.0,0.0,-0.5)),
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::K)) 
            => GameCommand::CameraMove(Vector3::new(0.0,0.0,0.5)),
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::W)) 
            => GameCommand::AvatorMoveW,
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::S)) 
            => GameCommand::AvatorMoveS,
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::A)) 
            => GameCommand::AvatorMoveA,
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::D)) 
            => GameCommand::AvatorMoveD,
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::Z)) 
            => GameCommand::CameraMove(Vector3::new(0.0,0.5,0.0)),
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::X)) 
            => GameCommand::CameraMove(Vector3::new(0.0,-0.5,0.0)),
        _   => { GameCommand::NOP}
    }
}

struct Image {
    data: Vec<u8>,
    width: u16,
    height: u16
}

gfx_defines!{
    vertex Vertex {
        pos: [f32; 3] = "position",
        normal: [f32; 3] = "normal",
        uv: [f32; 2] = "uv",
        joint_indices: [i32; 4] = "joint_indices",
        joint_weights: [f32; 4] = "joint_weights",
    }
    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        u_model_view_proj: gfx::Global<[[f32; 4]; 4]> = "u_model_view_proj",
        u_model_view: gfx::Global<[[f32; 4]; 4]> = "u_model_view",
        u_light: gfx::Global<[f32; 3]> = "u_light",
        u_ambient_color: gfx::Global<[f32; 4]> = "u_ambientColor",
        u_eye_direction: gfx::Global<[f32; 3]> = "u_eyeDirection",
        u_texture: gfx::TextureSampler<[f32; 4]> = "u_texture",
        u_translate: gfx::Global<[[f32; 4]; 4]> = "u_translate",
        out: gfx::RenderTarget<ColorFormat> = "Target0",
        out_depth: gfx::DepthTarget<gfx::format::DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

struct Camera<T> {
    pos: Point3<T>,
    target: Point3<T>,
    // up: Vector3<T>,
    view: Matrix4<T>,
    perspective: Matrix4<T>,
    projection: Matrix4<T>
}

impl<T: cgmath::BaseFloat> Camera<T> {
    fn new(pos: Point3<T>, target: Point3<T>, perspective: cgmath::PerspectiveFov<T>) -> Camera<T> {
        let view = Matrix4::look_at( pos
                                   , Point3::origin()
                                   , Vector3::new(Zero::zero(), Zero::zero(), One::one()));
        let pers = Matrix4::from(perspective);
        Camera {
            pos: pos,
            target: target,
            view: view,
            perspective: pers,
            projection: pers * view
        }
    }
    fn translate(&mut self, v: Vector3<T>) {
        self.pos += v;
    }
    fn lookTo(&mut self, target: Point3<T>) {
        self.target = target;
    }
    fn direction(& self) -> Vector3<T> {
        self.target - self.pos
    }
    fn update(&mut self) {
        self.view = Matrix4::look_at(self.pos
                                    ,self.target
                                    ,Vector3::new(Zero::zero(), Zero::zero(), One::one()));
        self.projection = self.perspective * self.view;
    }
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4]
        }
    }
}

const CLEAR_COLOR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];


pub struct Entry<R: gfx::Resources> {
    slice: gfx::Slice<R>,
    vertex_buffer: gfx::handle::Buffer<R, Vertex>,
    position: Vector3<f32>,
    front: Vector3<f32>,
}

fn vtn_to_vertex(a: collada::VTNIndex, obj: &collada::Object) -> Vertex {
    let mut vertex: Vertex = Default::default();
    let position = obj.vertices[a.0];
    vertex.pos = [position.x as f32, position.y as f32, position.z as f32];

    if obj.joint_weights.len() == obj.vertices.len() {
        let weights = obj.joint_weights[a.0];
        vertex.joint_weights = weights.weights;
        vertex.joint_indices = [
            weights.joints[0] as i32,
            weights.joints[1] as i32,
            weights.joints[2] as i32,
            weights.joints[3] as i32,
        ];
    }

    if let Some(uv) = a.1 {
        let uv = obj.tex_vertices[uv];
        vertex.uv = [uv.x as f32, uv.y as f32];
    }
    if let Some(normal) = a.2 {
        let normal = obj.normals[normal];
        vertex.normal = [normal.x as f32, normal.y as f32, normal.z as f32];
    }
    vertex
}

fn register_collada(conn: &mut Connection, object_id: i32, collada_name: &str, texture_id: i32, texture_name: &str) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;

    let img = open_texture(&std::path::Path::new(texture_name));
    let collada_doc = ColladaDocument::from_path(&Path::new(collada_name)).expect("failed to load dae");
    let collada_objs = collada_doc.get_obj_set().expect("cannot read obj set");

    insert_object(&tx, object_id, &collada_name, texture_id).expect("failed to insert sqlite (Object)");
    insert_texture(&tx, texture_id, &texture_name, img).expect("failed to insert sqlite (Texture)");

    for (i, obj) in collada_objs.objects.iter().enumerate() {
        register_collada_object(&tx, &obj, object_id, i as i32 + 1)
    }
    tx.commit()
}

fn register_collada_object(tx: &rusqlite::Transaction, obj: &collada::Object, object_id: i32, mesh_id: i32) {
    let mut i = 0;
    insert_mesh(&tx, object_id, mesh_id, &obj.name).expect("failed to insert sqlite (Mesh)");
    for geom in obj.geometry.iter() {
       let mut add = |a: collada::VTNIndex| {
           i += 1;
           insert_vertex(&tx, object_id, mesh_id, &vtn_to_vertex(a, obj), i).ok()
       };
       for shape in geom.shapes.iter() {
           match shape {
               &collada::Shape::Triangle(a, b, c) => {
                   add(a);
                   add(b);
                   add(c);
               }
               _ => {}
           }
       }
    }
}

fn create_table(conn: &Connection) {
   conn.execute("
CREATE TABLE Object 
  ( ObjectId    INTEGER PRIMARY KEY,
    Name        TEXT NOT NULL,
    TextureId   INTEGER NOT NULL
  );
", &[]).expect("create failure1"); 
   conn.execute("
CREATE TABLE Mesh 
  ( ObjectId  INTEGER NOT NULL,
    MeshId    INTEGER NOT NULL,
    Name      TEXT NOT NULL,
    PRIMARY KEY (ObjectId, MeshId)
  );", &[]).expect("create failure2"); 
   conn.execute("
CREATE TABLE Texture
  ( TextureId  INTEGER NOT NULL,
    Name      TEXT NOT NULL,
    Width     INTEGER NOT NULL,
    Height    INTEGER NOT NULL,
    Data   Blob NOT NULL,
    PRIMARY KEY (TextureId)
  );", &[]).expect("create failure3"); 
   conn.execute("
CREATE TABLE MeshVertex 
  ( ObjectId      INTEGER NOT NULL,
    MeshId        INTEGER NOT NULL, 
    IndexNo         INTEGER NOT NULL,
    PositionX     REAL NOT NULL,
    PositionY     REAL NOT NULL,
    PositionZ     REAL NOT NULL,
    NormalX       REAL NOT NULL,
    NormalY       REAL NOT NULL,
    NormalZ       REAL NOT NULL,
    U             REAL NOT NULL,
    V             REAL NOT NULL,
    Joint1        INTEGER NOT NULL,
    Joint2        INTEGER NOT NULL,
    Joint3        INTEGER NOT NULL,
    Joint4        INTEGER NOT NULL,
    JointWeight1  REAL NOT NULL,
    JointWeight2  REAL NOT NULL,
    JointWeight3  REAL NOT NULL,
    JointWeight4  REAL NOT NULL,
    PRIMARY KEY (ObjectId, MeshId, IndexNo)
  );
", &[]).expect("create failure4"); 
}

fn insert_object(tx: &rusqlite::Transaction, object_id: i32, name: &str, texture_id: i32) -> Result<i32, rusqlite::Error> {
   let mut stmt = tx.prepare("
INSERT INTO Object 
  (ObjectId, Name, TextureId) 
VALUES 
  ($1, $2, $3);
").expect("failed to insert Object");
   stmt.execute(&[&object_id, &name, &texture_id])
}
fn insert_texture(tx: &rusqlite::Transaction, texture_id: i32, name: &str, img: Image) -> Result<i32, rusqlite::Error> {
   let mut stmt = tx.prepare("
INSERT INTO Texture
  ( TextureId  
  , Name     
  , Width
  , Height
  , Data  
  )
VALUES 
  ($1, $2, $3, $4, $5);
").expect("failed to insert Texture");
   stmt.execute(&[&texture_id, &name, &(img.width as i32), &(img.height as i32), &img.data])
}

fn insert_mesh(tx: &rusqlite::Transaction, object_id: i32, mesh_id: i32, name: &str) -> Result<i32, rusqlite::Error> {
   let mut stmt = tx.prepare("
INSERT INTO Mesh 
  ( ObjectId
  , MeshId  
  , Name    
  )
VALUES 
  ($1, $2, $3);
").expect("failed to insert Mesh");
   stmt.execute(&[&object_id, &mesh_id, &name])
}

fn insert_vertex(tx: &rusqlite::Transaction, object_id: i32, mesh_id: i32, v: &Vertex, inx: i32) -> Result<i32, rusqlite::Error> {

   let mut stmt = tx.prepare("
INSERT INTO MeshVertex 
  ( ObjectId     ,
    MeshId       ,
    IndexNo      ,
    PositionX    ,
    PositionY    ,
    PositionZ    ,
    NormalX      ,
    NormalY      ,
    NormalZ      ,
    U            ,
    V            ,
    Joint1       ,
    Joint2       ,
    Joint3       ,
    Joint4       ,
    JointWeight1 ,
    JointWeight2 ,
    JointWeight3 ,
    JointWeight4 )
VALUES
  ($1 ,$2 ,$3 ,$4 ,$5 ,$6 ,$7 ,$8 ,$9 ,$10 ,$11 ,$12 ,$13 ,$14 ,$15 ,$16 ,$17 ,$18 ,$19)
").expect("failed to insert MeshVertex");
   stmt.execute(&[&object_id, &mesh_id, &inx,
                  &(v.pos[0] as f64), &(v.pos[1] as f64), &(v.pos[2] as f64),
                  &(v.normal[0] as f64), &(v.normal[1] as f64), &(v.normal[2] as f64),
                  &(v.uv[0] as f64), &(v.uv[1] as f64),
                  &0,&0,&0,&0,
                  &0,&0,&0,&0])
}

fn query_mesh(conn: &Connection, object_id: i32) -> Vec<Vec<Vertex>> {
   let mut stmt = conn.prepare("
SELECT 
  M.MeshId
, MV.PositionX   
, MV.PositionY   
, MV.PositionZ   
, MV.NormalX     
, MV.NormalY     
, MV.NormalZ     
, MV.U           
, MV.V           
, MV.Joint1      
, MV.Joint2      
, MV.Joint3      
, MV.Joint4      
, MV.JointWeight1
, MV.JointWeight2
, MV.JointWeight3
, MV.JointWeight4
  FROM Object AS O
LEFT JOIN Mesh AS M
  ON O.ObjectId = M.ObjectId
LEFT JOIN MeshVertex AS MV
  ON M.ObjectId = MV.ObjectId
  and M.MeshId = MV.MeshId
WHERE O.ObjectId = $1
Order By MV.ObjectId, MV.MeshId, MV.IndexNo
").expect("sql failure:"); 
   let mut meshes = Vec::new();
   let result = stmt.query_map(&[&object_id], |r| {
       ( r.get::<&str,i32>("MeshId") as usize
       , Vertex { 
            pos: [ r.get::<&str,f64>("PositionX") as f32
                 , r.get::<&str,f64>("PositionY") as f32
                 , r.get::<&str,f64>("PositionZ") as f32],
            normal: [ r.get::<&str,f64>("NormalX") as f32
                    , r.get::<&str,f64>("NormalY") as f32
                    , r.get::<&str,f64>("NormalZ") as f32],
            uv: [ r.get::<&str,f64>("U") as f32
                , r.get::<&str,f64>("V") as f32],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4]
        })
   }).expect("query failure");
   for r in result 
   {
       let (mesh_id, v) = r.expect("wrap failure");
       if meshes.len() < mesh_id 
       { 
           meshes.push(Vec::new());
       }
       meshes[mesh_id - 1].push(v);
   }
   meshes
}
fn query_texture(conn: &Connection, object_id: i32) -> Image {
   let mut stmt = conn.prepare("
SELECT 
  T.Width
, T.Height
, T.Data
  FROM Object AS O
LEFT JOIN Texture AS T
  ON O.TextureId = T.TextureId
WHERE O.ObjectId = $1
").expect("select failure"); 
   let result = stmt.query_map(&[&object_id], |r| {
       Image {
           data: r.get::<&str, Vec<u8>>("Data"),
           width: r.get::<&str, i32>("Width") as u16, 
           height: r.get::<&str, i32>("Height") as u16
       }
   }).expect("select failure");
   result.last().expect("select1").expect("select2")
}
fn open_texture(path: &std::path::Path) -> Image
{
    use std::io::BufReader;
    use png;
    let fin = std::fs::File::open(path).expect("no such file");
    let fin = BufReader::new(fin);
    let dec = png::Decoder::new(fin);
    let (_, mut reader) = dec.read_info().expect("collada load failure");
    // let color = reader.output_color_type().into();
    let mut data = vec![0; reader.output_buffer_size()];
    reader.next_frame(&mut data).ok();
    let (w, h) = reader.info().size(); 

    Image {
        data: data,
        width: w as u16,
        height: h as u16
    }
}

