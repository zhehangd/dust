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
