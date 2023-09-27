# import os

import sys
from atomate.vasp.builders.dielectric import DielectricBuilder
from atomate.vasp.builders.fix_tasks import FixTasksBuilder
from atomate.vasp.builders.tags import TagsBuilder
from atomate.vasp.builders.tasks_materials import TasksMaterialsBuilder


def run_builder(db_file: str):
    build_sequence = [
        FixTasksBuilder,
        TasksMaterialsBuilder,
        TagsBuilder,
        DielectricBuilder,
    ]
    for cls in build_sequence:
        b = cls.from_file(db_file)
        b.run()


if __name__ == "__main__":
    run_builder(sys.argv[1])
