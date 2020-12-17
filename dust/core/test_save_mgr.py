import os

from dust import _dust

from dust.core import save_mgr

def test_save_manager_create():
    with _dust.create_temporary_project() as proj:
        proj.parse_args()
        saver = save_mgr.SaveManager(project=proj)

def test_save_manager_simple():
    T = 10
    major_T = 35
    with _dust.create_temporary_project() as proj:
        proj.parse_args()
        saver = save_mgr.SaveManager(
            project=proj,
            minor_save_interval=T,
            major_save_interval=major_T)
        assert saver.next_save_tick() == 10
        saver.save(10, {})
        assert saver.next_save_tick() == 20
        saver.save(19, {})
        assert saver.next_save_tick() == 20
        saver.save(21, {})
        assert saver.next_save_tick() == 30
        saver.save(31, {})
        assert saver.next_save_tick() == 40
        saver.save(55, {})
        assert saver.next_save_tick() == 60
        
def test_find_saves():
    tick_list = [10, 19, 21, 31, 55, 111, 119, 127, 169, 200]
    timestamp1 = '2020-12-16-01-02-03'
    timestamp2 = '2020-12-16-02-50-00'
    timestamp3 = '2020-12-16-03-00-59'
    # Save two sequences, and break the second one with two timestamps
    # Should be able to retrieve the second sequence.
    with _dust.create_temporary_project(timestamp=timestamp1) as proj:
        proj.parse_args()
        save_kwargs = dict(project=proj, minor_save_interval=10,
                        major_save_interval=1000)
        
        saver = save_mgr.SaveManager(**save_kwargs)
        assert timestamp1 == proj.timestamp
        for i in range(6):
            saver.save(tick_list[i], {})
        
        proj.renew_timestamp(timestamp2)
        assert timestamp2 == proj.timestamp
        saver = save_mgr.SaveManager(**save_kwargs)
        for i in range(6):
            saver.save(tick_list[i], {})
        
        proj.renew_timestamp(timestamp3)
        assert timestamp3 == proj.timestamp
        saver = save_mgr.SaveManager(**save_kwargs)
        for i in range(6, len(tick_list)):
            saver.save(tick_list[i], {})
        assert len(os.listdir(saver.save_dir)) == 16
        
        saver = save_mgr.SaveManager(**save_kwargs)
        assert saver.scan_saves() == len(tick_list)
        
        save_list = saver.get_save_list()
        assert len(save_list) == len(tick_list)
        assert timestamp3 in save_list[-1]
        assert str(tick_list[-1]) in save_list[-1]

def test_save_cleanup():
    with _dust.create_temporary_project() as proj:
        proj.parse_args()
        save_kwargs = dict(
            project=proj, minor_save_interval=10, major_save_interval=35)
        saver = save_mgr.SaveManager(**save_kwargs)
        def test_save(save_tick, num_after):
            name = saver.save(save_tick, {})
            save_list = saver.get_save_list()
            assert len(save_list) == num_after
            assert save_list[-1] == name
        test_save(10, 1)
        test_save(19, 2)
        test_save(21, 3)
        test_save(31, 4)
        test_save(55, 2)
        test_save(64, 3)
        test_save(100, 3)
        test_save(105, 4)
        test_save(106, 5)
        test_save(110, 6)
        test_save(300, 5)
        test_save(600, 6)
        
def test_save_cleanup2():
    with _dust.create_temporary_project() as proj:
        proj.parse_args()
        save_kwargs = dict(
            project=proj, minor_save_interval=10, major_save_interval=35)
        saver = save_mgr.SaveManager(**save_kwargs)
        def test_save(save_tick, num_after):
            name = saver.save(save_tick, {})
            save_list = saver.get_save_list()
            assert len(save_list) == num_after
            assert save_list[-1] == name
        test_save(99, 1)
        test_save(100, 2)
        test_save(106, 2)
        














