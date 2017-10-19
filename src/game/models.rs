use std;
use rusqlite::Connection;
use rusqlite::Error as RusqliteError;
use cgmath::{
    Matrix4,
};

#[derive(Debug, Copy, Clone)]
pub struct Joint {
    pub joint_index: i32,
    pub global : Matrix4<f32>,
    pub bind: Matrix4<f32>,
    pub parent: i32,
    pub inverse: Matrix4<f32>
}

#[derive(Debug)]
pub struct Animation {
    pub joint_index: i32,
    pub time: f32,
    pub pose: Matrix4<f32>,
}

pub struct Image<T> {
    pub data: Vec<u8>,
    pub width: u16,
    pub height: u16,
    pub format: std::marker::PhantomData<T>
}

pub type RusqliteResult<T> = Result<T, RusqliteError>;

pub fn query_animation(conn: &Connection, object_id: &i32) -> RusqliteResult<Vec<Vec<(f32, Animation)>>> {
    let mut stmt = conn.prepare("
SELECT
    AnimationId ,
    ObjectId    ,
    JointIndex  ,
    SampleTime  ,
    SamplePose11,
    SamplePose12,
    SamplePose13,
    SamplePose14,
    SamplePose21,
    SamplePose22,
    SamplePose23,
    SamplePose24,
    SamplePose31,
    SamplePose32,
    SamplePose33,
    SamplePose34,
    SamplePose41,
    SamplePose42,
    SamplePose43,
    SamplePose44,
    Name        
  FROM Animation AS A
WHERE A.ObjectId = ?1
  AND JointIndex <> 0
Order By JointIndex, SampleTime
")?;
    let result = stmt.query_map(&[object_id], |r| {
        ( r.get::<&str,i32>("AnimationId"),
          r.get::<&str,i32>("JointIndex"),
          r.get::<&str,f64>("SampleTime") as f32,
          Matrix4::new(r.get::<&str,f64>("SamplePose11") as f32,
                       r.get::<&str,f64>("SamplePose12") as f32,
                       r.get::<&str,f64>("SamplePose13") as f32,
                       r.get::<&str,f64>("SamplePose14") as f32,
                       r.get::<&str,f64>("SamplePose21") as f32,
                       r.get::<&str,f64>("SamplePose22") as f32,
                       r.get::<&str,f64>("SamplePose23") as f32,
                       r.get::<&str,f64>("SamplePose24") as f32,
                       r.get::<&str,f64>("SamplePose31") as f32,
                       r.get::<&str,f64>("SamplePose32") as f32,
                       r.get::<&str,f64>("SamplePose33") as f32,
                       r.get::<&str,f64>("SamplePose34") as f32,
                       r.get::<&str,f64>("SamplePose41") as f32,
                       r.get::<&str,f64>("SamplePose42") as f32,
                       r.get::<&str,f64>("SamplePose43") as f32,
                       r.get::<&str,f64>("SamplePose44") as f32)
        )
    })?;

    let mut animations = Vec::<Vec<(f32, Animation)>>::with_capacity(255);
    for r in result
    {
        let (id, joint_index, time, pose) = r?;

        if joint_index >= 0 {
            (|t: (f32, Animation) | {
                if match animations.get(joint_index as usize) { Some(_) => true, _ => false } {
                    animations.get_mut(joint_index as usize).unwrap().push(t);
                } else {
                    for i in animations.len() .. joint_index as usize {
                        animations.insert(i, vec!()); 
                    }
                    animations.insert(joint_index as usize, vec!(t)); 
                }
            })((time,
                Animation {
                    joint_index,
                    time,
                    pose,
                })
              );
        }
    }
    Ok(animations)
}
