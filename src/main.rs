#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate collada;
extern crate cgmath;
extern crate rusqlite;

use gfx::traits::FactoryExt;
use gfx::Device;

use std::path::Path;
use collada::document::ColladaDocument;
use rusqlite::Connection;

pub type ColorFormat = gfx::format::Rgba8;
pub type DepthFormat = gfx::format::DepthStencil;
use rusqlite::types::ToSql;

use cgmath:: {
    EuclideanSpace,
    Point3,
    Vector3,
    Matrix4,
    One,
    Zero,
    // Rotation,
    // Quaternion,
};


pub fn main() {
    // let mut conn = Connection::open(&Path::new("test.sqlite3")).unwrap();
    let mut conn = Connection::open_in_memory().unwrap();
    create_table(&conn);
    register_collada(&mut conn, 1, "assets/house.dae").unwrap();



    let width = 1024;
    let height = 768;
    let builder = glutin::WindowBuilder::new()
        .with_title("PARTI")
        .with_dimensions(width, height)
        .with_vsync();
    let (window, mut device, mut factory, main_color, main_depth) =
        gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder);

    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();
    let pso = factory.create_pipeline_simple(
        include_bytes!("shader/150.glslv"),
        include_bytes!("shader/150.glslf"),
        pipe::new()
    ).unwrap();

    let mut entries = Vec::new();

    let result = query_mesh(&conn, 1);
    for vertex_data in result
    {
        let index_data: Vec<u32> = vertex_data.iter().enumerate().map(|(i, _)| i as u32).collect();
        let (vbuf, slice) = factory.create_vertex_buffer_with_slice(&vertex_data, &index_data[..]);

        entries.push(Entry {
            slice: slice,
            vertex_buffer: vbuf
        });
    }
    let mut camera = Camera::new(Point3::new(0.0, -7.0, 0.0),
                                 Point3::origin(),
                                 cgmath::PerspectiveFov  {
                                     fovy: cgmath::Rad(120.0f32.to_radians()),
                                     aspect: (width as f32) / (height as f32),
                                     near: 0.001,
                                     far: 1000.0,
                                 });

    'main: loop {
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::L)) 
                    => { camera.move_pos(Vector3::new(-0.5,0.0,0.0)); },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::H)) 
                    => { camera.move_pos(Vector3::new(0.5,0.0,0.0)); },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::J)) 
                    => { camera.move_pos(Vector3::new(0.0,0.0,0.5)); },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::K)) 
                    => { camera.move_pos(Vector3::new(0.0,0.0,-0.5)); },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::Z)) 
                    => { camera.move_pos(Vector3::new(0.0,0.5,0.0)); },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(glutin::VirtualKeyCode::X)) 
                    => { camera.move_pos(Vector3::new(0.0,-0.5,0.0)); },
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
                out: main_color.clone(),
                out_depth: main_depth.clone()
            };
            encoder.draw(&entry.slice, &pso, &data);
        }
        encoder.flush(&mut device);
        window.swap_buffers().unwrap();
        device.cleanup();
    }
}

gfx_defines!{
    vertex Vertex {
        pos: [f32; 3] = "position",
        normal: [f32; 3] = "normal",
        uv: [f32; 2] = "uv",
        color: [f32; 3] = "a_Color",
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
    fn move_pos(&mut self, v: Vector3<T>){
        self.pos += v;
        self.view = Matrix4::look_at(self.pos
                                    ,self.target
                                    ,Vector3::new(Zero::zero(), Zero::zero(), One::one()));
        self.projection = self.perspective * self.view;
    }
    fn direction(& self) -> Vector3<T>{
        self.target - self.pos
    }
}

impl Default for Vertex {
    fn default() -> Vertex {
        Vertex {
            pos: [0.0; 3],
            normal: [0.0; 3],
            uv: [0.0; 2],
            color: [0.0; 3],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4]
        }
    }
}

const CLEAR_COLOR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];

pub struct Entry<R: gfx::Resources> {
    slice: gfx::Slice<R>,
    vertex_buffer: gfx::handle::Buffer<R, Vertex>
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

fn register_collada(conn: &mut Connection, object_id: i32, collada_name: &str) -> rusqlite::Result<()> {
    let tx = try!(conn.transaction());
    let collada_doc = ColladaDocument::from_path(&Path::new(collada_name)).unwrap();
    let collada_objs = collada_doc.get_obj_set().unwrap();

    insert_object(&tx, object_id, "human", &collada_name);

    for (i, obj) in collada_objs.objects.iter().enumerate() {
        register_collada_object(&tx, &obj, object_id, i as i32 + 1)
    }
    tx.commit()
}

fn register_collada_object(tx: &rusqlite::Transaction, obj: &collada::Object, objectId: i32, mesh_id: i32) {
    let mut i = 0;
    insert_mesh(&tx, objectId, mesh_id, &obj.name);
    for geom in obj.geometry.iter() {
       let mut add = |a: collada::VTNIndex| {
           i += 1;
           insert_vertex(&tx, objectId, mesh_id, &vtn_to_vertex(a, obj), i);
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
    ColladaFile TEXT NOT NULL
  );
", &[]).unwrap(); 
   conn.execute("
CREATE TABLE Mesh 
  ( ObjectId  INTEGER NOT NULL,
    MeshId    INTEGER NOT NULL,
    Name      TEXT NOT NULL,
    Texture   TEXT,
    DiffuseR  REAL NOT NULL,
    DiffuseG  REAL NOT NULL,
    DiffuseB  REAL NOT NULL,
    DiffuseA  REAL NOT NULL,
    PRIMARY KEY (ObjectId, MeshId)
  );", &[]).unwrap(); 
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
", &[]).unwrap(); 
}

fn insert_object(tx: &rusqlite::Transaction, object_id: i32, name: &str, file: &str) -> Result<i32, rusqlite::Error> {
   let mut stmt = tx.prepare("
INSERT INTO Object 
  (ObjectId, Name, ColladaFile) 
VALUES 
  ($1, $2, $3);
").unwrap();
   stmt.execute(&[&object_id, &name, &file])
}
fn insert_mesh(tx: &rusqlite::Transaction, object_id: i32, mesh_id: i32, name: &str) -> Result<i32, rusqlite::Error> {
   let mut stmt = tx.prepare("
INSERT INTO Mesh 
  ( ObjectId
  , MeshId  
  , Name    
  , Texture 
  , DiffuseR
  , DiffuseG
  , DiffuseB
  , DiffuseA
  )
VALUES 
  ($1, $2, $3, $4, $5, $6, $7, $8);
").unwrap();
    let color = match name {
        "eye-mesh" => [0.0,0.0,0.0],
        "mayu-mesh" => [0.1149353,0.03830004,0.004554826],
        "hair_h-mesh" => [0.1149353,0.03830004,0.004554826],
        "hair-mesh" => [0.1149353,0.03830004,0.004554826],
        "body-mesh" => [0.8,0.525,0.23],
        _ => {println!("{}", name); [0.5; 3]}
    };

   stmt.execute(&[&object_id, &mesh_id, &name, &"", &color[0], &color[1], &color[2], &1])
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
").unwrap();
   stmt.execute(&[&object_id, &mesh_id, &inx,
                  &(v.pos[0] as f64) as &ToSql, &(v.pos[1] as f64) as &ToSql, &(v.pos[2] as f64) as &ToSql,
                  &(v.normal[0] as f64) as &ToSql, &(v.normal[1] as f64) as &ToSql, &(v.normal[2] as f64) as &ToSql,
                  &(v.uv[0] as f64) as &ToSql, &(v.uv[1] as f64) as &ToSql,
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
, M.DiffuseR
, M.DiffuseG
, M.DiffuseB
, M.DiffuseA
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
").unwrap(); 
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
            color: [ r.get::<&str,f64>("DiffuseR") as f32
                   , r.get::<&str,f64>("DiffuseG") as f32
                   , r.get::<&str,f64>("DiffuseB") as f32],
            joint_indices: [0; 4],
            joint_weights: [0.0; 4]
        })
   }).unwrap();
   for r in result 
   {
       let (mesh_id, v) = r.unwrap();
       if meshes.len() < mesh_id 
       { 
           meshes.push(Vec::new());
       }
       meshes[mesh_id - 1].push(v);
   }
   meshes
}


