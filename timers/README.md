These are systemd timers I run on my personal server to update some things.

I really would prefer to run these as GitHub Actions jobs, but it turns out that GitHub
Actions scheduled jobs don't run anywhere near the scheduled time, and when that time is
within the hour or two after 0:00 UTC, often don't run at all.

This is no good. So back to timers on personal servers.

The scripts that post to reddit and twitter need secrets and API keys etc, they source
them as environment variables from a file `secrets.sh` one directory up from the main
repository, these are not stored publicly in the repository for obvious reasons.

Notes for how to make and enable timers:
========================================

A basic systemd unit file to be triggered on a timer looks like:

```
#foo.service

[Unit]
Description=Foo

[Service]
ExecStart=/path/to/foo
User=<user>
```

```
#foo.timer

[Unit]
Description=Timer for Foo

[Timer]
OnCalendar=*-*-* <HH:MM:SS>

[Install]
WantedBy=timers.target

```

To enable:

```
sudo systemctl link $PWD/foo.service
sudo systemctl enable --now $PWD/foo.timer
```
to disable:
```
sudo systemctl disable foo.timer
sudo systemctl disable foo.service
```

to manually run a job (with the .service file already linked)
```
sudo systemctl start foo
```

to stop a job that's currently running
```
sudo systemctl stop foo
```

to reload after changing the unit file(s):
```
sudo systemctl daemon-reload
```

to list timers and see when they ran last/will run next:
```
systemctl list-timers
```

to view logs for a unit (last 1000 lines + live updating):
```
journalctl -u foo.service -n 1000 -f
```
