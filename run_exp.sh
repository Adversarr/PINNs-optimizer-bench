if [ ! -d results ]; then
  mkdir results
fi

python main.py --optim=muon --init-lr=8e-2 --adamw-lr=4e-3 --sched=exp --epoch=1000
python main.py --optim=muon --init-lr=4e-2 --adamw-lr=2e-3 --sched=exp --epoch=1000
python main.py --optim=muon --init-lr=2e-2 --adamw-lr=1e-3 --sched=exp --epoch=1000
python main.py --optim=muon --init-lr=1e-2 --adamw-lr=5e-4 --sched=exp --epoch=1000

python main.py --optim=adam --init-lr=4e-3 --sched=exp --epoch=1000
python main.py --optim=adam --init-lr=2e-3 --sched=exp --epoch=1000
python main.py --optim=adam --init-lr=1e-3 --sched=exp --epoch=1000
python main.py --optim=adam --init-lr=8e-3 --sched=exp --epoch=1000
