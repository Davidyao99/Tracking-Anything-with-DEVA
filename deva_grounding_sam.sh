#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <parent_directory>"
    exit 1
fi

# Store the parent directory and Python script in variables
parent_directory="$1"


# Check if the parent directory is a valid directory
if [ ! -d "$parent_directory" ]; then
    echo "$parent_directory is not a valid directory."
    exit 1
fi

# Use the find command to locate all directories in the parent directory
# and pass each directory as an argument to the Python script
find "$parent_directory" -mindepth 1 -maxdepth 1 -type d | sort | while read -r directory; do
    if [ $((counter % $2)) -eq  $3 ]; then
    # Call the Python script with the directory as an argument
    echo "working on $directory"
    fbname=$(basename "$directory")
    echo "saving to $fbname"
    # python3 demo/demo_with_text.py --chunk_size 4 --img_path "$directory/color" --amp --DINO_THRESHOLD 0.3 --DINO_NMS_THRESHOLD 0.7 --temporal_setting semionline --max_missed_detection_count 1 --size -1 --output "$directory/deva_0.3" --prompt "table.dryer.wardrobe.book.cooler.dispenser.container.detergent.cup.fireplace.light.ball.laptop.kettle.ladder.plate.board.oven.dresser.person.bucket.pitcher.powerstrip.plants.washing machine.stairs.toaster.jacket.nightstand.bookshelf.suitcase.blackboard.hat.box.cabinet.sign.poster.ledge.bathtub.tv.outlet.printer.chair.rack.window.fan.ottoman.toilet.counter.vacuum.copier.keyboard.projector.screen.shoe.picture.rail.alarm.closet.stool.cutter.vent.tube.mattress.divider.hair dryer.mouse.door.bed.dishwasher.mat.switch.calendar.mailbox.bar.decoration.clothes.broom.basket.machine.stove.dumbbell.headphones.seat.furniture.plunger.doorframe.curtain.column.rod.pillow.sink.extinguisher.toys.telephone.tissue.ceiling.whiteboard.coffee maker.microwave.refrigerator.paper.dustpan.crate.plant.bowl.backpack.piano.mirror.guitar.blinds.tray.bench.blanket.windowsill.bottle.towel.clock.pipe.armchair.cushion.dustbin.speaker.stand.monitor.lamp.pillar.radiator.bag"
#    python3 demo/demo_with_text.py --chunk_size 4 --img_path "$directory/color" --amp --DINO_THRESHOLD 0.35 --DINO_NMS_THRESHOLD 0.7 --temporal_setting semionline --max_missed_detection_count 1 --size -1 --output "$directory/deva" --prompt "wall.chair.floor.table.door.couch.cabinet.shelf.desk.office chair.bed.pillow.sink.picture.window.toilet.bookshelf.monitor.curtain.book.armchair.coffee table.box.refrigerator.lamp.kitchen cabinet.towel.clothes.tv.nightstand.counter.dresser.stool.cushion.plant.ceiling.bathtub.end table.dining table.keyboard.bag.backpack.toilet paper.printer.tv stand.whiteboard.blanket.shower curtain.trash can.closet.stairs.microwave.stove.shoe.computer tower.bottle.bin.ottoman.bench.board.washing machine.mirror.copier.basket.sofa chair.file cabinet.fan.laptop.shower.paper.person.paper towel dispenser.oven.blinds.rack.plate.blackboard.piano.suitcase.rail.radiator.recycling bin.container.wardrobe.soap dispenser.telephone.bucket.clock.stand.light.laundry basket.pipe.clothes dryer.guitar.toilet paper holder.seat.speaker.column.ladder.bathroom stall.shower wall.cup.jacket.storage bin.coffee maker.dishwasher.paper towel roll.machine.mat.windowsill.bar.toaster.bulletin board.ironing board.fireplace.soap dish.kitchen counter.doorframe.toilet paper dispenser.mini fridge.fire extinguisher.ball.hat.shower curtain rod.water cooler.paper cutter.tray.shower door.pillar.ledge.toaster oven.mouse.toilet seat cover dispenser.furniture.cart.scale.tissue box.light switch.crate.power outlet.decoration.sign.projector.closet door.vacuum cleaner.plunger.stuffed animal.headphones.dish rack.broom.range hood.dustpan.hair dryer.water bottle.handicap bar.vent.shower floor.water pitcher.mailbox.bowl.paper bag.projector screen.divider.laundry detergent.bathroom counter.object.bathroom vanity.closet wall.laundry hamper.bathroom stall door.ceiling light.trash bin.dumbbell.stair rail.tube.bathroom cabinet.closet rod.coffee kettle.shower head.keyboard piano.case of water bottles.coat rack.folded chair.fire alarm.power strip.calendar.poster.potted plant.mattress"
    python3 demo/demo_with_text.py --chunk_size 4 --gsam --img_path "$directory/color" --amp --DINO_THRESHOLD 0.35 --DINO_NMS_THRESHOLD 0.7 --temporal_setting semionline --max_missed_detection_count 1 --size -1 --output "$directory/deva" --prompt_file "$directory/prompt.txt"
    fi
    ((counter++))
done