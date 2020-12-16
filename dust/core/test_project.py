import gc
import os

from dust.core import project

def check_project(proj):
    assert proj.proj_dir
    assert os.path.isdir(proj.proj_dir)
    assert proj.log_filename
    assert os.path.isfile(proj.log_filename)
    assert proj.log_filename.startswith(proj.proj_dir)

def test_create_temporary_project():
    proj = project.create_temporary_project()
    check_project(proj)
    proj_dir = proj.proj_dir
    proj.release()
    assert not os.path.isdir(proj_dir), proj_dir

def test_detach_global_project():
    project.detach_global_project()
    proj = project.create_temporary_project()
    project.detach_global_project()
    proj = project.create_temporary_project()
    project.detach_global_project()


def test_project_context():
    with project.create_temporary_project() as proj:
        pass
